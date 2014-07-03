RUSTC = rustc -O src/lib.rs

.PHONY: all bench clean test

all:
	cargo build

bench:
	mkdir -p bin
	$(RUSTC) $(LTO) --test --out-dir bin
	RUST_THREADS=1 bin/linalg --bench --ratchet-metrics metrics.json

clean:
	rm -rf lib

test:
	cargo test
	./check-line-length.sh
