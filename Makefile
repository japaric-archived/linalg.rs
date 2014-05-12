RUSTC = rustc -O src/lib.rs
#LTO = -Z lto

.PHONY: all bench clean test

all:
	mkdir -p lib
	$(RUSTC) --out-dir lib

bench:
	mkdir -p bin
	$(RUSTC) $(LTO) --test --out-dir bin
	RUST_THREADS=1 bin/linalg --bench --ratchet-metrics metrics.json

clean:
	rm -rf bin
	rm -rf lib

test:
	mkdir -p bin
	$(RUSTC) $(LTO) --test --out-dir bin
	RUST_THREADS=1 bin/linalg
	./check-line-length.sh
