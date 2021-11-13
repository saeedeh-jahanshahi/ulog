CFLAGS = -O2 -Wall -mavx512f -mavx -mavx512bw
CC = gcc
LIBS = -lm

object-files = convert_avx512.o log_operation_avx512.o

%: %.c convert_avx512.h $(object-files)
	$(CC) $(CFLAGS) $(object-files) $@.c -o $@ $(LIBS)
	@echo "Successfull compilation of $@"

convert_avx512.o: convert_avx512.h macro_avx512.h

log_operation_avx512.o: log_operation_avx512.h macro_avx512.h
