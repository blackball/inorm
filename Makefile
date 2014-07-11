all: inorm-test

inorm-test:
	clang inorm.c test.c -o inorm-test -lopencv_core -lopencv_highgui -lopencv_imgproc -lm

clean: 
	rm inorm-test
