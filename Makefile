TYPE ?= Release
TEST ?= ON
USE_CUDA ?= ON

CMAKE_OPT = -DBUILD_TEST=$(TEST)
CMAKE_OPT += -DBUILD_TESTING=OFF
CMAKE_OPT += -DUSE_CUDA=$(USE_CUDA)

build:
	mkdir -p build/$(TYPE)
	cd build/$(TYPE) && cmake $(CMAKE_OPT) ../.. && make -j

clean:
	rm -rf build

test-cpp:
	@echo
	cd build/$(TYPE) && make test
