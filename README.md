# vlm-from-scratch

Translation of <https://github.com/hkproj/pytorch-paligemma>

## Usage

clojure -M:run-inference --model-path ./paligemma-3b-pt-224 --prompt "describe the image" --image-file-path ~/Downloads/FIFA_22_Cover.jpg --do-sample --max-tokens 200
