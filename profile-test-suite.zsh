#!/bin/zsh

cargo clean

COMPILATION=$({ time cargo test ^$ >/dev/null } 2>&1)

COMPILE_CPU=$(echo $COMPILATION | cut -d ' ' -f 11)
COMPILE_TIME=$(echo $COMPILATION | cut -d ' ' -f 13)

EXECUTION=$({ time RUST_LOG=quickcheck cargo test 2>&1 } 2>&1)

EXECUTION_CPU=$(echo $EXECUTION | tail -n 1 | cut -d ' ' -f 10)
EXECUTION_TIME=$(echo $EXECUTION | tail -n 1 | cut -d ' ' -f 12)
UNIT_TESTS=$(echo $EXECUTION | grep passed | cut -d ' ' -f 4 | paste -sd+ | bc)
QC_TESTS=$(echo $EXECUTION | grep Passed | cut -d ' ' -f 3 | paste -sd+ | bc)
TEST_FILES=$(ls tests/*.rs | wc -l)

echo "Compiled $TEST_FILES test files in $COMPILE_TIME s using $COMPILE_CPU CPU"
echo "Executed $UNIT_TESTS unit tests ($QC_TESTS QC tests) in $EXECUTION_TIME s using $EXECUTION_CPU CPU"
