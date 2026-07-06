(ns vlm.inference
  (:require [libpython-clj2.require :refer [require-python]]
            [clj-pytorch.functional :as t]
            [clj-pytorch.inference :as infer]
            [clj-pytorch.context :as ctx]
            [clj-pytorch.tensor :as tensor]
            [clojure.tools.cli :as cli]
            [vlm.processing-paligemma :refer [make-processor process]]
            [vlm.utils :refer [load-hf-model]]))

(require-python '[PIL.Image :as pil-image])

(defn move-to-device [inputs device]
  (into {} (map (fn [[k v]] [k (t/to-device v device)]) inputs)))

(defn get-model-inputs [processor prompt image-path device]
  (let [image  (pil-image/open image-path)
        inputs (process processor [prompt] [image])]
    (move-to-device inputs device)))

(defn test-inference
  [model processor device prompt image-path
   & {:keys [max-tokens temperature top-p do-sample]
      :or   {max-tokens 100 temperature 0.8 top-p 0.9 do-sample false}}]
  (let [inputs    (get-model-inputs processor prompt image-path device)
        tokenizer (:tokenizer processor)
        stop-id   (infer/eos-token-id tokenizer)]
    (loop [state     inputs
           generated []
           remaining max-tokens]
      (if (zero? remaining)
        (println (str prompt (infer/tokenizer-decode tokenizer (t/tensor (vec generated)))))
        (let [py-out     (ctx/no-grad (model (:input-ids state)
                                             (:pixel-values state)
                                             (:attention-mask state)))
              logits     (t/select py-out 1 -1)
              next-token (ctx/inference-mode
                           (infer/sample-next-token logits
                                                    :do-sample do-sample
                                                    :temperature temperature
                                                    :top-p top-p))
              token-id   (t/item (t/squeeze next-token))
              generated  (conj generated token-id)]
          (if (= token-id stop-id)
            (println (str prompt (infer/tokenizer-decode tokenizer (t/tensor (vec generated)))))
            (recur
             (assoc state
                    :input-ids      next-token
                    :attention-mask (infer/extend-attention-mask (:attention-mask state)))
             generated
             (dec remaining))))))))

(def ^:private cli-options
  [[nil "--model-path PATH"      "Path to the model checkpoint directory"]
   [nil "--prompt TEXT"          "Text prompt"]
   [nil "--image-file-path PATH" "Path to the image file"]
   [nil "--max-tokens N"         "Max tokens to generate"    :default 100  :parse-fn #(Integer/parseInt %)]
   [nil "--temperature T"        "Sampling temperature"      :default 0.8  :parse-fn #(Double/parseDouble %)]
   [nil "--top-p P"              "Top-p nucleus sampling"    :default 0.9  :parse-fn #(Double/parseDouble %)]
   [nil "--do-sample"            "Use sampling (vs greedy)"  :default false]
   [nil "--only-cpu"             "Force CPU inference"       :default false]
   ["-h" "--help"]])

(defn -main [& args]
  (let [{:keys [options errors summary]} (cli/parse-opts args cli-options)
        {:keys [model-path prompt image-file-path max-tokens
                temperature top-p do-sample only-cpu help]} options]
    (cond
      help   (println summary)
      errors (do (run! println errors) (System/exit 1))
      (not (and model-path prompt image-file-path))
      (do (println "Required: --model-path --prompt --image-file-path")
          (println summary)
          (System/exit 1))
      :else
      (let [device                          (if only-cpu "cpu" (tensor/best-device))
            _                               (println "Device in use:" device)
            {:keys [model tokenizer config]} (load-hf-model model-path device)
            processor                       (-> (make-processor tokenizer
                                                               (-> config :text-config :num-image-tokens)
                                                               (-> config :vision-config :image-size))
                                                (assoc :config config))]
        (println "Running inference")
        (test-inference model processor device prompt image-file-path
                        :max-tokens  max-tokens
                        :temperature temperature
                        :top-p       top-p
                        :do-sample   do-sample)
        (System/exit 0)))))
