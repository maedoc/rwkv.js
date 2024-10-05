rwkv.js: rwkv-bind.cpp libggml.wasm
	emcc -O3 -msimd128 -I rwkv.cpp/ggml/include rwkv-bind.cpp rwkv.cpp/rwkv.cpp libggml.wasm -lm -lembind -o rwkv.js

libggml.wasm:
	cd rwkv.cpp/ggml && CFLAGS="-msimd128" cmake -B build -S . -DCMAKE_C_COMPILER=emcc -DCMAKE_CXX_COMPILER=emcc -DCMAKE_C_FLAGS="-msimd128"  -DCMAKE_CXX_FLAGS="-msimd128" && cd build && make -j || true
	cp rwkv.cpp/ggml/build/src/libggml.so $@