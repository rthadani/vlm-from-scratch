(ns vlm.modeling-gemma
  (:require [libpython-clj2.python :refer [py. py.-] :as py]
            [libpython-clj2.require :refer [require-python]]
            [clj-pytorch.nn :as nn :refer [defmodule]]
            [clj-pytorch.functional :as t]
            [clj-pytorch.tensor :as tensor]
            [clj-pytorch.context :as ctx]
            [vlm.modeling-siglip :refer [SiglipVisionModel siglip-config]]))

(require-python '[builtins :as builtins])

(defn gemma-config
  [& {:keys [vocab-size hidden-size intermediate-size num-hidden-layers
             num-attention-heads num-key-value-heads head-dim
             max-position-embeddings rms-norm-eps rope-theta
             attention-bias attention-dropout pad-token-id]
      :or {head-dim 256 max-position-embeddings 8192 rms-norm-eps 1e-6
           rope-theta 10000.0 attention-bias false attention-dropout 0.0}}]
  {:vocab-size vocab-size
   :hidden-size hidden-size
   :intermediate-size intermediate-size
   :num-hidden-layers num-hidden-layers
   :num-attention-heads num-attention-heads
   :num-key-value-heads num-key-value-heads
   :head-dim head-dim
   :max-position-embeddings max-position-embeddings
   :rms-norm-eps rms-norm-eps
   :rope-theta rope-theta
   :attention-bias attention-bias
   :attention-dropout attention-dropout
   :pad-token-id pad-token-id})

(defn paligemma-config
  [& {:keys [vision-config text-config ignore-index image-token-index
             projection-dim hidden-size pad-token-id]
      :or {ignore-index -100 image-token-index 256000
           projection-dim 2048 hidden-size 2048}}]
  (let [v-cfg (apply siglip-config (mapcat identity vision-config))
        t-cfg (apply gemma-config (concat (mapcat identity text-config)
                                          [:pad-token-id pad-token-id]))
        num-patches (int (Math/pow (/ (:image-size v-cfg) (:patch-size v-cfg)) 2))]
    {:ignore-index ignore-index
     :image-token-index image-token-index
     :vocab-size (:vocab-size t-cfg)
     :projection-dim projection-dim
     :hidden-size hidden-size
     :pad-token-id pad-token-id
     :is-encoder-decoder false
     :vision-config (assoc v-cfg :projection-dim projection-dim)
     :text-config (assoc t-cfg :num-image-tokens num-patches)}))

(defmodule GemmaRMSNorm
  [dim eps]
  :layers {}
  :init (fn [self]
          (py/set-attr! self "eps" eps)
          (nn/register-parameter! self :weight (t/zeros [dim])))
  :forward (fn [self x]
             (let [weight (py.- self weight)
                   eps (py.- self eps)
                   xf (tensor/->float x)
                   norm (-> xf (t/pow 2) (t/mean -1 true) (t/add eps) t/rsqrt (t/mul xf))
                   output (t/mul norm (t/add 1.0 (tensor/->float weight)))]
               (t/type-as output x))))

(defmodule GemmaRotaryEmbedding
  [dim max-position-embeddings base]
  :layers {}
  :init (fn [self]
          (let [angles (-> (t/arange 0 dim 2 t/int64)
                           tensor/->float
                           (t/div (float dim)))
                inv-f (t/div 1.0 (t/scalar-pow (float base) angles))]
            (nn/register-buffer! self "inv_freq" inv-f :persistent false)))
  :forward (fn [self x position-ids _seq-len]
             (let [dev-type (let [dt (first (clojure.string/split (t/device-of x) #":"))]
                              (if (= dt "mps") "cpu" dt))
                   inv-freq (t/to-device (py.- self inv_freq) (t/device-of x))
                   batch-size (first (t/shape position-ids))
                   inv-freq-expanded (-> inv-freq
                                         (t/unsqueeze 0) ;[None, :, None] is Python fancy indexing — None at a position inserts a size-1 dimension there
                                         (t/unsqueeze -1)
                                         tensor/->float
                                         (t/expand [batch-size -1 1]))
                   position-ids-expanded (-> position-ids (t/unsqueeze 1) tensor/->float)]
               (ctx/no-autocast dev-type
                                (let [freqs (-> (t/matmul inv-freq-expanded position-ids-expanded) (t/transpose 1 2))
                                      emb (t/cat (builtins/list [freqs freqs]) :dim -1)]
                                  [(t/to-dtype (t/cos emb) (t/dtype x))
                                   (t/to-dtype (t/sin emb) (t/dtype x))])))))

(defn rotate-half [x]
  (let [[x1 x2] (t/chunk x 2 :dim -1)]
    (t/cat (builtins/list [(t/mul x2 -1) x1]) :dim -1)))

(defn apply-rotary-pos-emb [q k cos sin]
  (let [cos (t/unsqueeze cos 1)
        sin (t/unsqueeze sin 1)]
    [(t/add (t/mul q cos) (t/mul (rotate-half q) sin))
     (t/add (t/mul k cos) (t/mul (rotate-half k) sin))]))

(defmodule GemmaMLP
  [{:keys [hidden-size intermediate-size]}]
  :layers {:gate-proj (nn/linear hidden-size intermediate-size :bias false)
           :up-proj (nn/linear hidden-size intermediate-size :bias false)
           :down-proj (nn/linear intermediate-size hidden-size :bias false)}
  :forward (fn [self x]
             (let [gate (-> x ((nn/get-layer self :gate-proj)) (t/gelu :approximate "tanh"))
                   up ((nn/get-layer self :up-proj) x)]
               ((nn/get-layer self :down-proj) (t/mul gate up)))))

;;flash attention is simulated inneficient since the key heads are repeated in memory
;;otherwise we need cuda trickery to share key
(defn repeat-kv [hidden-states n-rep]
  (if (= n-rep 1)
    hidden-states
    (let [[batch num-kv-heads slen head-dim] (t/shape hidden-states)]
      (-> hidden-states
          (t/unsqueeze 2)
          (t/expand [batch num-kv-heads n-rep slen head-dim])
          (t/reshape [batch (* num-kv-heads n-rep) slen head-dim])))))

(defmodule GemmaAttention
  [{:keys [attention-dropout hidden-size num-attention-heads head-dim
           num-key-value-heads max-position-embeddings rope-theta
           attention-bias] :as config}
   layer-idx]
  :layers {:q-proj (nn/linear hidden-size (* num-attention-heads head-dim) :bias attention-bias)
           :k-proj (nn/linear hidden-size (* num-key-value-heads head-dim) :bias attention-bias)
           :v-proj (nn/linear hidden-size (* num-key-value-heads head-dim) :bias attention-bias)
           :o-proj (nn/linear (* num-attention-heads head-dim) hidden-size :bias attention-bias)
           :rotary-emb (GemmaRotaryEmbedding head-dim max-position-embeddings rope-theta)}
  :init (fn [self]
          (py/set-attr! self "key_cache" nil)
          (py/set-attr! self "value_cache" nil))
  :forward (fn [self hidden-states attention-mask position-ids]
             (let [num-key-value-groups (quot num-attention-heads num-key-value-heads)
                   [bsz q-len _] (t/size hidden-states)
                   query-states (-> hidden-states ((nn/get-layer self :q-proj))
                                    (t/view [bsz q-len num-attention-heads head-dim])
                                    (t/transpose 1 2))
                   key-states (-> hidden-states ((nn/get-layer self :k-proj))
                                  (t/view [bsz q-len num-key-value-heads head-dim])
                                  (t/transpose 1 2))
                   value-states (-> hidden-states ((nn/get-layer self :v-proj))
                                    (t/view [bsz q-len num-key-value-heads head-dim])
                                    (t/transpose 1 2))
                   [cos sin] ((nn/get-layer self :rotary-emb) value-states position-ids nil)
                   [query-states key-states] (apply-rotary-pos-emb query-states key-states cos sin)
                   k-cached (py.- self key_cache)
                   v-cached (py.- self value_cache)
                   key-states   (if (nil? k-cached) key-states
                                  (t/cat (builtins/list [k-cached key-states]) :dim -2))
                   value-states (if (nil? v-cached) value-states
                                  (t/cat (builtins/list [v-cached value-states]) :dim -2))
                   _            (py/set-attr! self "key_cache" key-states)
                   _            (py/set-attr! self "value_cache" value-states)
                   key-states (repeat-kv key-states num-key-value-groups)
                   value-states (repeat-kv value-states num-key-value-groups)
                   scale (/ 1.0 (Math/sqrt (double head-dim)))
                   attn-weights (-> (t/matmul query-states (t/transpose key-states 2 3))
                                    (t/mul scale)
                                    (t/add attention-mask))
                   attn-weights (-> attn-weights
                                    (t/softmax :dim -1 :dtype t/float32)
                                    (t/to-dtype (t/dtype query-states))
                                    (t/dropout-f attention-dropout (nn/training? self)))
                   attn-output (-> (t/matmul attn-weights value-states)
                                   (t/transpose 1 2)
                                   t/contiguous
                                   (t/view [bsz q-len -1])
                                   ((nn/get-layer self :o-proj)))]
               [attn-output attn-weights])))

(defmodule GemmaDecoderLayer
  [{:keys [hidden-size rms-norm-eps] :as config} layer-idx]
  :layers {:self-attn (GemmaAttention config layer-idx)
           :mlp (GemmaMLP config)
           :input-layernorm (GemmaRMSNorm hidden-size rms-norm-eps)
           :post-attention-layernorm (GemmaRMSNorm hidden-size rms-norm-eps)}
  :forward (fn [self hidden-states attention-mask position-ids]
             (let [residual hidden-states
                   hidden-states ((nn/get-layer self :input-layernorm) hidden-states)
                   [hidden-states _] ((nn/get-layer self :self-attn)
                                      hidden-states attention-mask position-ids)
                   hidden-states (t/add residual hidden-states)
                   residual hidden-states
                   hidden-states ((nn/get-layer self :post-attention-layernorm) hidden-states)
                   hidden-states ((nn/get-layer self :mlp) hidden-states)]
               (t/add residual hidden-states))))

(defmodule GemmaModel
  [{:keys [vocab-size hidden-size num-hidden-layers rms-norm-eps pad-token-id] :as config}]
  :layers {:embed-tokens (nn/embedding vocab-size hidden-size :padding-idx pad-token-id)
           :layers (nn/module-list (map #(GemmaDecoderLayer config %) (range num-hidden-layers)))
           :norm (GemmaRMSNorm hidden-size rms-norm-eps)}
  :forward (fn [self attention-mask position-ids inputs-embeds]
             (let [normalizer (t/tensor (Math/sqrt (double hidden-size))
                                        :dtype (t/dtype inputs-embeds))
                   hidden-states (t/mul inputs-embeds normalizer)
                   hidden-states (reduce (fn [hidden-states decoder-layer]
                                           (decoder-layer hidden-states attention-mask position-ids))
                                         hidden-states
                                         (nn/module-list-seq self :layers))]
               ((nn/get-layer self :norm) hidden-states))))

(defmodule GemmaForCausalLM
  [{:keys [vocab-size hidden-size] :as config}]
  :layers {:model (GemmaModel config)
           :lm-head (nn/linear hidden-size vocab-size :bias false)}
  :forward (fn [self attention-mask position-ids inputs-embeds]
             (let [hidden-states ((nn/get-layer self :model)
                                  attention-mask position-ids inputs-embeds)]
               (-> hidden-states ((nn/get-layer self :lm-head)) tensor/->float))))

(defn tie-weights!
  "Sets lm_head.weight = embed_tokens.weight (weight tying)."
  [causal-lm]
  (let [py-model (.py-module causal-lm)
        lm-head (py.- py-model lm_head)
        embed-toks (-> py-model (py.- model) (py.- embed_tokens))]
    (py/set-attr! lm-head "weight" (py.- embed-toks weight))))

(defmodule PaliGemmaMultiModalProjector
  [{:keys [vision-config]}]
  :layers {:linear (nn/linear (:hidden-size vision-config) (:projection-dim vision-config) :bias true)}
  :forward (fn [self image-features]
             ((nn/get-layer self :linear) image-features)))

(defn merge-input-ids-with-image-features
  [image-features inputs-embeds input-ids attention-mask n-cached
   {:keys [hidden-size image-token-index]} pad-token-id]
  (let [[_ _ embed-dim] (t/shape image-features)
        [batch-size seq-len] (t/shape input-ids)
        dtype (t/dtype inputs-embeds)
        device (t/device-of inputs-embeds)
        scaled-img (t/mul image-features (/ 1.0 (Math/sqrt (double hidden-size))))
        final-embedding (t/zeros [batch-size seq-len embed-dim] :dtype dtype :device device)
        text-mask (t/logical-and (t/ne input-ids image-token-index)
                                 (t/ne input-ids pad-token-id))
        image-mask (t/eq input-ids image-token-index)
        pad-mask (t/eq input-ids pad-token-id)
        text-exp (-> text-mask (t/unsqueeze -1) (t/expand [-1 -1 embed-dim]))
        pad-exp (-> pad-mask (t/unsqueeze -1) (t/expand [-1 -1 embed-dim]))
        image-exp (-> image-mask (t/unsqueeze -1) (t/expand [-1 -1 embed-dim]))
        final-embedding (as-> final-embedding $
                          (t/where text-exp inputs-embeds $)
                          (t/masked-scatter $ image-exp scaled-img)
                          (t/where pad-exp (t/zeros-like $) $))
        q-len (second (t/shape inputs-embeds))
        causal-mask (-> (t/full [batch-size q-len (+ n-cached q-len)]
                                0 :dtype dtype :device device)
                        (t/unsqueeze 1))
        position-ids (if (pos? n-cached)
                       (let [ids (-> attention-mask (t/cumsum -1) (t/select 1 -1))]
                         (if (= 1 (count (t/shape ids))) (t/unsqueeze ids 0) ids))
                       (-> (t/cumsum attention-mask -1)
                           (t/masked-fill! (t/eq attention-mask 0) 1)
                           (t/to-device device)))]
    [final-embedding causal-mask position-ids]))

(defmodule PaliGemmaForConditionalGeneration
  [{:keys [vision-config text-config pad-token-id] :as config}]
  :layers {:vision-tower (SiglipVisionModel vision-config)
           :multi-modal-projector (PaliGemmaMultiModalProjector config)
           :language-model (GemmaForCausalLM text-config)}
  :init (fn [self]
          (py/set-attr! self "n_cached" 0))
  :forward (fn [self input-ids pixel-values attention-mask]
             (let [n-cached     (py.- self n_cached)
                   embed-tokens (-> (nn/get-layer self :language-model)
                                    (nn/get-layer :model)
                                    (nn/get-layer :embed-tokens))
                   inputs-embeds (embed-tokens input-ids)
                   image-features (-> pixel-values
                                      (t/to-dtype (t/dtype inputs-embeds))
                                      (->> ((nn/get-layer self :vision-tower))))
                   image-features ((nn/get-layer self :multi-modal-projector) image-features)
                   [inputs-embeds
                    attention-mask
                    position-ids] (merge-input-ids-with-image-features
                                   image-features inputs-embeds input-ids attention-mask
                                   n-cached config (or pad-token-id -1))
                   logits ((nn/get-layer self :language-model)
                           attention-mask position-ids inputs-embeds)]
               (py/set-attr! self "n_cached" (+ n-cached (second (t/shape input-ids))))
               logits)))

(defn reset-cache! [model]
  (let [py-model (.py-module model)]
    (py/set-attr! py-model "n_cached" 0)
    (doseq [m (py. py-model modules)]
      (when (py/has-attr? m "key_cache")
        (py/set-attr! m "key_cache" nil)
        (py/set-attr! m "value_cache" nil)))))
