(ns vlm.processing-paligemma
  (:require [libpython-clj2.python :refer [py. py.-] :as py]
            [libpython-clj2.require :refer [require-python]]
            [clj-pytorch.functional :as t]
            [clj-pytorch.inference :as infer]
            [clj-pytorch.context :as ctx]))

(require-python '[numpy :as np])
(require-python '[PIL.Image :as pil-image])
(require-python '[torch :as torch])
(require-python '[builtins :as builtins])

(def imagenet-standard-mean [0.5 0.5 0.5])
(def imagenet-standard-std [0.5 0.5 0.5])

(def image-token "<image>")

(defn add-image-tokens-to-prompt [prefix-prompt bos-token image-seq-len img-token]
  (str (apply str (repeat image-seq-len img-token))
       bos-token
       prefix-prompt
       "\n"))

(defn rescale [image scale]
  (np/multiply image (float scale)))

(defn resize [image [height width] resample]
  (py. image resize [width height] :resample resample))

(defn normalize [image mean std]
  (let [dtype (py.- image dtype)
        mean (np/array mean :dtype dtype)
        std (np/array std :dtype dtype)]
    (np/divide (np/subtract image mean) std)))

(defn process-images [images {:keys [size resample rescale-factor image-mean image-std]}]
  (->> images
       (map #(resize % size resample))
       (map np/array)
       (map #(rescale % rescale-factor))
       (map #(normalize % image-mean image-std))
       (map #(py. % transpose [2 0 1]))
       (map t/tensor)
       builtins/list
       t/stack))

(defn make-processor [tokenizer num-image-tokens image-size]
  (let [extra-tokens (builtins/list
                      (mapv builtins/str
                            (concat [image-token]
                                    (map #(format "<loc%04d>" %) (range 1024))
                                    (map #(format "<seg%03d>" %) (range 128)))))]
    (py. tokenizer add_tokens extra-tokens :special_tokens true))
  (let [image-token-id (py. tokenizer convert_tokens_to_ids image-token)]
    (py/set-attr! tokenizer "add_bos_token" false)
    (py/set-attr! tokenizer "add_eos_token" false)
    {:tokenizer tokenizer
     :image-seq-len num-image-tokens
     :image-size image-size
     :image-token-id image-token-id}))

(defn process
  [{:keys [tokenizer image-seq-len image-size]} texts images
   & {:keys [padding truncation] :or {padding "longest" truncation true}}]
  (assert (= 1 (count images) (count texts)))
  (let [pixel-values (ctx/inference-mode
                      (process-images images
                                      {:size [image-size image-size]
                                       :resample (py.- pil-image/Resampling BICUBIC)
                                       :rescale-factor (/ 1.0 255.0)
                                       :image-mean imagenet-standard-mean
                                       :image-std imagenet-standard-std}))
        input-strings (builtins/list
                       (mapv #(builtins/str (add-image-tokens-to-prompt
                                             % (py.- tokenizer bos_token) image-seq-len image-token))
                             texts))
        inputs (infer/tokenizer-encode tokenizer input-strings
                                       :return-tensors "pt"
                                       :padding padding
                                       :truncation truncation)]
    (assoc inputs :pixel-values pixel-values)))
