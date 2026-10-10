CLANG_FORMAT ?= clang-format

.PHONY: format
format:
	$(CLANG_FORMAT) -i data_loader/cpp/*.cpp data_loader/cpp/*.h tests/*.cpp
