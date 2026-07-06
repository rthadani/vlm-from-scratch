(ns vlm.utils
  (:require [libpython-clj2.python :refer [py. py.-] :as py]
            [libpython-clj2.require :refer [require-python]]
            [clojure.data.json :as json]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [clj-pytorch.nn :as nn]
            [vlm.modeling-gemma :refer [PaliGemmaForConditionalGeneration paligemma-config]]))

(require-python '[safetensors.torch :as safetensors-torch])
(require-python '[builtins :as builtins])

(defn- read-config [model-path]
  (json/read-str (slurp (io/file model-path "config.json"))
                 :key-fn (fn [k] (-> k (str/replace "_" "-") keyword))))

(defn- load-weights [model-path]
  (let [files   (->> (file-seq (io/file model-path))
                     (filter #(str/ends-with? (.getName ^java.io.File %) ".safetensors")))
        tensors (builtins/dict)]
    (doseq [f files]
      (py. tensors update (safetensors-torch/load_file (str f) :device "cpu")))
    tensors))

(defn- load-tokenizer [model-path]
  (let [xf (py/import-module "transformers")]
    (py. (py.- xf AutoTokenizer) from_pretrained model-path :padding_side "right")))

(defn load-hf-model
  "Load a PaliGemma HuggingFace checkpoint from model-path onto device.
   Returns {:model :tokenizer :config :device}."
  [model-path device]
  (let [tokenizer (load-tokenizer model-path)
        raw-cfg   (read-config model-path)
        config    (paligemma-config
                    :ignore-index      (:ignore-index raw-cfg -100)
                    :image-token-index (:image-token-index raw-cfg 256000)
                    :projection-dim    (:projection-dim raw-cfg 2048)
                    :hidden-size       (:hidden-size raw-cfg 2048)
                    :pad-token-id      (:pad-token-id raw-cfg)
                    :vision-config     (:vision-config raw-cfg)
                    :text-config       (:text-config raw-cfg))
        weights   (load-weights model-path)
        model     (PaliGemmaForConditionalGeneration config)
        py-m      (.py-module model)]
    (py/call-attr-kw py-m "load_state_dict" [weights] {:strict false})
    (py. py-m to device)
    (nn/eval! model)
    (let [lm (py.- py-m language_model)]
      (py/set-attr! (py.- lm lm_head) "weight"
                    (-> lm (py.- model) (py.- embed_tokens) (py.- weight))))
    {:model model :tokenizer tokenizer :config config :device device}))
