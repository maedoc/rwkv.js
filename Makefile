rwkv.js: rwkv-bind.cpp rwkv.wasm libggml.wasm
	emcc rwkv-bind.cpp rwkv.wasm libggml.wasm -lm -lembind --preload-file rwkv6.q41 -sALLOW_MEMORY_GROWTH=1

rwkv.wasm:
	emcc -O3 -msimd128 -I rwkv.cpp/ggml/include -c rwkv.cpp/rwkv.cpp
	mv rwkv.o rwkv.wasm

libggml.wasm:
	cd rwkv.cpp/ggml && cmake -B build -S . -DCMAKE_C_COMPILER=emcc -DCMAKE_CXX_COMPILER=emcc -DCMAKE_C_FLAGS="-msimd128"  -DCMAKE_CXX_FLAGS="-msimd128" -DGGML_BUILD_TESTS=false -DGGML_BUILD_EXAMPLES=false && cd build && make -j || true
	cp rwkv.cpp/ggml/build/src/libggml.so $@

env:
	python3 -m venv env
	. env/bin/activate
	python -m pip install tokenizers 'numpy<2'
	python -m pip install torch  --index-url https://download.pytorch.org/whl/cpu

rwkv6-f16.pth:
	curl -L 'https://huggingface.co/BlinkDL/rwkv-6-world/resolve/main/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth?download=true' > $@

rwkv6.q41: rwkv6-f16.pth rwkv.cpp/librwkv.so
	python rwkv.cpp/python/convert_pytorch_to_ggml.py rwkv6-f16.pth rwkv6.f16 FP16
	python rwkv.cpp/python/quantize.py rwkv6.f16 $@ Q4_1

rwkv.cpp/librwkv.so:
	. env/bin/activate
	cd rwkv.cpp && cmake . && cmake --build . --config Release
