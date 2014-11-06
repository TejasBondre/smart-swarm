OBJ = swarm.o
INC = -I "./"

swarm: $(OBJ)
	g++ -L/usr/lib $(OBJ) -lglut -lGL -lGLU -o a.out
	rm -f $(OBJ)

main.o:
	g++ -c swarm.cpp $(INC)

clean:
	rm -f $(OBJ) swarm