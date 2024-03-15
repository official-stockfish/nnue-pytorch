SRCS = training_data_loader.cpp
HEADERS = lib/nnue_training_data_formats.h lib/nnue_training_data_stream.h lib/rng.h

format:
	black .
	clang-format -i $(SRCS) $(HEADERS) -style=file