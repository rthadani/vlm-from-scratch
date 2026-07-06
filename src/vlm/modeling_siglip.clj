(ns vlm.modeling-siglip
  (:require [libpython-clj2.python :refer [py.-]]
            [libpython-clj2.require :refer [require-python]]
            [clj-pytorch.nn :as nn :refer [defmodule]]
            [clj-pytorch.functional :as t]
            [clj-pytorch.tensor :as tensor]))

(require-python '[torch :as torch])

(defn siglip-config
  [& {:keys [hidden-size intermediate-size num-hidden-layers
             num-attention-heads num-channels image-size patch-size
             layer-norm-eps attention-dropout]
      :or {hidden-size 768
           intermediate-size 3072
           num-hidden-layers 12
           num-attention-heads 12
           num-channels 3
           image-size 224
           patch-size 16
           layer-norm-eps 1e-6
           attention-dropout 0.0}}]
  (let [num-patches (int (Math/pow (/ image-size patch-size) 2))]
    {:hidden-size hidden-size
     :intermediate-size intermediate-size
     :num-hidden-layers num-hidden-layers
     :num-attention-heads num-attention-heads
     :num-channels num-channels
     :image-size image-size
     :patch-size patch-size
     :layer-norm-eps layer-norm-eps
     :attention-dropout attention-dropout
     :num-patches num-patches
     :head-dim (/ hidden-size num-attention-heads)}))

(defmodule SiglipVisionEmbeddings
  [config]
  :init (fn [self]
          (nn/register-buffer! self "position_ids"
                               (-> (:num-patches config)
                                   (t/arange)
                                   (t/expand [1 -1]))
                               :persistent false))
  :layers {:patch-embedding (nn/conv2d (:num-channels config)
                                       (:hidden-size config)
                                       (:patch-size config)
                                       :stride (:patch-size config)
                                       :padding "valid")
           :position-embedding (nn/embedding (:num-patches config)
                                             (:hidden-size config))}
  :forward (fn [self pixel-values]
             (let [embeddings (-> pixel-values
                                  ((nn/get-layer self :patch-embedding))
                                  (t/flatten 2)
                                  (t/transpose 1 2))
                   pos-ids (py.- self "position_ids")
                   pos-embeds ((nn/get-layer self :position-embedding) pos-ids)]
               (t/add embeddings pos-embeds))))

(defmodule SiglipAttention
  [{:keys [hidden-size] :as config}]
  :layers {:k-proj (nn/linear hidden-size hidden-size)
           :v-proj (nn/linear hidden-size hidden-size)
           :q-proj (nn/linear hidden-size hidden-size)
           :out-proj (nn/linear hidden-size hidden-size)}
  :forward (fn [self hidden-states]
             (let [embed-dim (:hidden-size config)
                   num-heads (:num-attention-heads config)
                   head-dim (:head-dim config)
                   scale (Math/pow head-dim -0.5)
                   [batch-size seq-len] (t/size hidden-states)
                   query-states ((nn/get-layer self :q-proj) hidden-states)
                   key-states ((nn/get-layer self :k-proj) hidden-states)
                   value-states ((nn/get-layer self :v-proj) hidden-states)
                   query-states (-> (t/view query-states [batch-size seq-len num-heads head-dim])
                                    (t/transpose 1 2))
                   key-states (-> (t/view key-states [batch-size seq-len num-heads head-dim])
                                  (t/transpose 1 2))
                   value-states (-> (t/view value-states [batch-size seq-len num-heads head-dim])
                                    (t/transpose 1 2))
                   attn-weights (-> (t/matmul query-states (t/transpose key-states 2 3))
                                    (t/mul scale)
                                    (t/softmax :dim -1 :dtype torch/float32)
                                    (t/to-dtype (t/dtype query-states))
                                    (t/dropout-f (:attention-dropout config) (nn/training? self)))
                   attn-output (-> (t/matmul attn-weights value-states)
                                   (t/transpose 1 2)
                                   (t/contiguous)
                                   (t/reshape [batch-size seq-len embed-dim])
                                   ((nn/get-layer self :out-proj)))]
               [attn-output attn-weights])))

(defmodule SiglipMLP
  [config]
  :layers {:fc1 (nn/linear (:hidden-size config) (:intermediate-size config))
           :fc2 (nn/linear (:intermediate-size config) (:hidden-size config))}
  :forward (fn [self hidden-states]
             (-> hidden-states
                 ((nn/get-layer self :fc1))
                 (t/gelu :approximate "tanh")
                 ((nn/get-layer self :fc2)))))

(defmodule SiglipEncoderLayer
  [config]
  :layers {:self-attn (SiglipAttention config)
           :layer-norm1 (nn/layer-norm (:hidden-size config) :eps (:layer-norm-eps config))
           :mlp (SiglipMLP config)
           :layer-norm2 (nn/layer-norm (:hidden-size config) :eps (:layer-norm-eps config))}
  :forward (fn [self hidden-states]
             (let [residual hidden-states
                   hidden-states ((nn/get-layer self :layer-norm1) hidden-states)
                   [hidden-states _] ((nn/get-layer self :self-attn) hidden-states)
                   hidden-states (t/add residual hidden-states)
                   residual hidden-states
                   hidden-states ((nn/get-layer self :layer-norm2) hidden-states)
                   hidden-states ((nn/get-layer self :mlp) hidden-states)]
               (t/add residual hidden-states))))

(defmodule SiglipEncoder
  [config]
  :layers {:layers (nn/module-list (repeatedly (:num-hidden-layers config)
                                               #(SiglipEncoderLayer config)))}
  :forward (fn [self inputs-embeds]
             (reduce (fn [hidden-states layer]
                       (layer hidden-states))
                     inputs-embeds
                     (nn/module-list-seq self :layers))))

(defmodule SiglipVisionTransformer
  [config]
  :layers {:embeddings (SiglipVisionEmbeddings config)
           :encoder (SiglipEncoder config)
           :post-layernorm (nn/layer-norm (:hidden-size config) :eps (:layer-norm-eps config))}
  :forward (fn [self pixel-values]
             (-> pixel-values
                 ((nn/get-layer self :embeddings))
                 ((nn/get-layer self :encoder))
                 ((nn/get-layer self :post-layernorm)))))

(defmodule SiglipVisionModel
  [config]
  :layers {:vision-model (SiglipVisionTransformer config)}
  :forward (fn [self pixel-values]
             ((nn/get-layer self :vision-model) pixel-values)))
