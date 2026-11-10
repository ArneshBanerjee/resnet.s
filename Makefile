SRC := src/main.s src/load.s src/model.s src/gemm.s src/math.s
OBJ := $(SRC:src/%.s=build/%.o)
SDK := $(shell xcrun --show-sdk-path)
HF  := https://huggingface.co/timm/resnet18.tv_in1k/resolve/main

resnet: $(OBJ)
	ld -arch arm64 -e _main -o $@ $(OBJ) -L$(SDK)/usr/lib -lSystem

build/%.o: src/%.s src/macros.inc
	@mkdir -p build
	as -arch arm64 -I src $< -o $@

# The original Hugging Face file. No conversion: resnet reads it as it is.
weights: model.safetensors
model.safetensors:
	curl -fL -o $@ $(HF)/$@

clean:
	rm -rf build resnet

.PHONY: weights clean
