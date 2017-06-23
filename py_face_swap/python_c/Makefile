CC= g++

# opencv_PATH = /usr/local
CFLAGS=  -I/usr/include/python2.7 -I/usr/include -fPIC 
# LIBFLAGS=-L$(opencv_PATH)/lib -lboost_python -L/usr/lib/python2.7/config -lpython2.7 -L/usr/lib/x86_64-linux-gnu/ -I$(opencv_PATH)/include -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_gpu -lopencv_objdetect -lopencv_ml 

# SOURCES=my.cpp
# OBJECTS=$(SOURCES:.cpp=.o)

my.so: my.o
	$(CC) -shared -Wl,--export-dynamic -o $@ $^ $(LIBFLAGS)   -lboost_python -L/usr/lib/python2.7/config -lpython2.7 -L/usr/lib/x86_64-linux-gnu/ 

.cpp.o:
	$(CC) $(CFLAGS) -c $< 

clean:
	rm -f *.o *.so