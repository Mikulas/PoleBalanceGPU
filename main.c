/**
 * @author Mikuláš Dítě
 * @license Original BSD, see license.txt
 */

/**
 * If the application fails during run time, the kernel could not be found in path and you
 *  1) must click on the file under "Executables" list item, then get info (or apple-I), then
 *  2) set the working directory to Project Directory (not build directory)
 */


#include <stdio.h>
#include <assert.h>
#include <sys/sysctl.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <OpenCL/OpenCL.h>


#pragma mark Utilities
char * load_program_source(const char *filename)
{
	struct stat statbuf;
	FILE *fh; 
	char *source; 

	fh = fopen(filename, "r");
	if (fh == 0)
		return 0; 

	stat(filename, &statbuf);
	source = (char *) malloc(statbuf.st_size + 1);
	fread(source, statbuf.st_size, 1, fh);
	source[statbuf.st_size] = '\0'; 

	return source; 
}

/** globals */
cl_program program[1];
cl_kernel kernel[2];
cl_command_queue cmd_queue;
cl_context context;
cl_device_id cpu = NULL, device = NULL;
cl_int err = 0;
size_t returned_size = 0;
size_t buffer_size;
cl_mem mem_c_position, mem_c_velocity, mem_p_angle, mem_p_velocity, mem_fitness;
int initiated = 0;

#pragma mark -
#pragma mark OpenCL context
int initGPU(int n)
{
	#pragma mark Device Information
	// Find the CPU CL device, as a fallback
	err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_CPU, 1, &cpu, NULL);
	assert(err == CL_SUCCESS);

	// Find the GPU CL device, this is what we really want
	// If there is no GPU device is CL capable, fall back to CPU
	err |= clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	if (err != CL_SUCCESS) device = cpu;
	assert(device);

	// Get some information about the returned device
	cl_char vendor_name[1024] = {0};
	cl_char device_name[1024] = {0};
	err |= clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(vendor_name), vendor_name, &returned_size);
	err |= clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, &returned_size);
	assert(err == CL_SUCCESS);
	printf("Connecting to %s %s...", vendor_name, device_name);

	#pragma mark Context and Command Queue
	// Now create a context to perform our calculation with the 
	// specified device 
	context = clCreateContext(0, 1, &device, NULL, NULL, &err);
	assert(err == CL_SUCCESS);

	// And also a command queue for the context
	cmd_queue = clCreateCommandQueue(context, device, 0, NULL);

	#pragma mark Program and Kernel Creation
	// Load the program source from disk
	// The kernel/program is the project directory and in Xcode the executable
	// is set to launch from that directory hence we use a relative path
	const char * filename = "kernel.cl";
	char *program_source = load_program_source(filename);
	program[0] = clCreateProgramWithSource(context, 1, (const char**)&program_source, NULL, &err);
	assert(err == CL_SUCCESS);

	err |= clBuildProgram(program[0], 0, NULL, NULL, NULL, NULL);
	assert(err == CL_SUCCESS);

	// Now create the kernel "objects" that we want to use in the example file 
	kernel[0] = clCreateKernel(program[0], "add", &err);
	assert(err == CL_SUCCESS);

	#pragma mark Memory Allocation
	// Allocate memory on the device to hold our data and store the results into
	buffer_size = sizeof(int) * n;

	mem_c_position = clCreateBuffer(context, CL_MEM_READ_ONLY, buffer_size, NULL, &err);
	mem_c_velocity = clCreateBuffer(context, CL_MEM_READ_ONLY, buffer_size, NULL, &err);
	mem_p_angle = clCreateBuffer(context, CL_MEM_READ_ONLY, buffer_size, NULL, &err);
	mem_p_velocity = clCreateBuffer(context, CL_MEM_READ_ONLY, buffer_size, NULL, &err);
	assert(err == CL_SUCCESS);

	mem_fitness = clCreateBuffer(context, CL_MEM_WRITE_ONLY, buffer_size, NULL, &err);
	assert(err == CL_SUCCESS);

	// Get all of the stuff written and allocated
	clFinish(cmd_queue);

	printf(" done\n");

	return err; // CL_SUCCESS
}

void terminateGPU()
{
	#pragma mark Teardown
	clReleaseMemObject(mem_c_position);
	clReleaseMemObject(mem_c_velocity);
	clReleaseMemObject(mem_p_angle);
	clReleaseMemObject(mem_p_velocity);
	clReleaseMemObject(mem_fitness);

	clReleaseCommandQueue(cmd_queue);
	clReleaseContext(context);
}

#pragma mark -
#pragma mark Generation context
int computeFitness(int * c_position, int * c_velocity, int * p_angle, int * p_velocity, int * fitness, int n)
{
	if (!initiated) {
		initGPU(n);
		initiated = 1;
	}

	#pragma mark Writing memory
	// Allocate memory on the device to hold our data and store the results into
	buffer_size = sizeof(int) * n;

	err = clEnqueueWriteBuffer(cmd_queue, mem_c_position, CL_TRUE, 0, buffer_size, (void *) c_position, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(cmd_queue, mem_c_velocity, CL_TRUE, 0, buffer_size, (void *) c_velocity, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(cmd_queue, mem_p_angle, CL_TRUE, 0, buffer_size, (void *) p_angle, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(cmd_queue, mem_p_velocity, CL_TRUE, 0, buffer_size, (void *) p_velocity, 0, NULL, NULL);
	assert(err == CL_SUCCESS);

	// Get all of the stuff written and allocated
	clFinish(cmd_queue);

	#pragma mark Kernel Arguments
	// Now setup the arguments to our kernel
	err  = clSetKernelArg(kernel[0], 0, sizeof(cl_mem), (void *) &mem_c_position);
	err |= clSetKernelArg(kernel[0], 1, sizeof(cl_mem), (void *) &mem_c_velocity);
	err |= clSetKernelArg(kernel[0], 2, sizeof(cl_mem), (void *) &mem_p_angle);
	err |= clSetKernelArg(kernel[0], 3, sizeof(cl_mem), (void *) &mem_p_velocity);
	err |= clSetKernelArg(kernel[0], 4, sizeof(cl_mem), (void *) &mem_fitness);
	assert(err == CL_SUCCESS);

	#pragma mark Execution and Reading memory
	// Run the calculation by enqueuing it and forcing the 
	// command queue to complete the task
	size_t global_work_size = n;
	err = clEnqueueNDRangeKernel(cmd_queue, kernel[0], 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
	assert(err == CL_SUCCESS);
	clFinish(cmd_queue);

	// Once finished read back the results from the answer 
	// array into the results array
	err = clEnqueueReadBuffer(cmd_queue, mem_fitness, CL_TRUE, 0, buffer_size, fitness, 0, NULL, NULL);
	assert(err == CL_SUCCESS);
	clFinish(cmd_queue);

	return CL_SUCCESS;
}

#pragma mark -
int main (int argc, const char * argv[]) {
	#pragma mark Configuration
	const int generation_size = 40;
	const int generation_count = 10000;
	const float mutation = 0.1;
	const int time_total = 60000; // should be the same as in kernel.cl

	srand(time(NULL));

	#pragma mark Allocate standard memory
	int * c_position = (int *) malloc(generation_size * sizeof(int));
	int * c_velocity = (int *) malloc(generation_size * sizeof(int));
	int * p_angle = (int *) malloc(generation_size * sizeof(int));
	int * p_velocity = (int *) malloc(generation_size * sizeof(int));
	int * fitness = (int *) malloc(generation_size * sizeof(int));
	int fitness_sum = 0;
	int best_key = 0;

	int * next_c_position = (int *) malloc(generation_size * sizeof(int));
	int * next_c_velocity = (int *) malloc(generation_size * sizeof(int));
	int * next_p_angle = (int *) malloc(generation_size * sizeof(int));
	int * next_p_velocity = (int *) malloc(generation_size * sizeof(int));

	#pragma mark Generate first generation
	for (int i = 0; i < generation_size; i++) {
		int sign = rand() % 2 == 1 ? 1 : -1;
		next_c_position[i] = sign * rand() % 1000;
		next_c_velocity[i] = sign * rand() % 1000;
		next_p_angle[i] = sign * rand() % 1000;
		next_p_velocity[i] = sign * rand() % 1000;
		// fitness[i] = 0;
	}

	#pragma mark Genetical algorithm
	int n;
	int last_sum = 0;
	for (n = 0; n < generation_count; n++) {
		c_position = next_c_position;
		c_velocity = next_c_velocity;
		p_angle = next_p_angle;
		p_velocity = next_p_velocity;

		fitness_sum = 0;
		best_key = 0;

		computeFitness(c_position, c_velocity, p_angle, p_velocity, fitness, generation_size);

		// prevent computing generation in the last cycle
		if (n == generation_count - 1) break;

		int fitness_max = 0;
		// TODO: allocate it only once
		int * border = (int *) malloc(generation_size * sizeof(int));
		for (int i = 0; i < generation_size; i++) {
			fitness_sum += fitness[i];
			if (fitness[i] > fitness_max) {
				fitness_max = fitness[i];
				best_key = i;
			}

			if (i == 0) {
				border[i] = fitness[i];
			} else {
				border[i] = border[i - 1] + fitness[i];
			}
		}
		// break if best solution is already found
		if (fitness_max >= time_total) break;

		//printf("gen[%d] best_fitness = \t%d\t[%d]\t%s\n", n, fitness_max, fitness_sum, last_sum < fitness_sum ? "up" : "FALLS");
		last_sum = fitness_sum;
		// Elite - always copy the best one
		next_c_position[0] = c_position[best_key];
		next_c_velocity[0] = c_velocity[best_key];
		next_p_angle[0] = p_angle[best_key];
		next_p_velocity[0] = p_velocity[best_key];

		for (int k = 1; k < generation_size; k++) {			
			int key_parent_1 = 0;
			int key_parent_2 = 0;

			// Get weighted entity (roulette wheel implementation)
			int roll = rand() % fitness_sum;
			int i;
			for (i = 0; i < generation_size; i++) {
				if (roll < border[i]) {
					break;
				}
			}
			key_parent_1 = i;

			roll = rand() % fitness_sum;
			for (i = 0; i < generation_size; i++) {
				if (roll < border[i]) {
					break;
				}
			}
			key_parent_2 = i;

			printf("%d\t", c_position[k]);

			// Prepare next generation as combination of two parens, with mutation
			next_c_position[k] = c_position[key_parent_1] + mutation * (rand() % 2 == 1 ? 1 : -1) * (rand() % (c_position[key_parent_1] == 0 ? 1 : c_position[key_parent_1]));
			next_c_velocity[k] = c_velocity[key_parent_1] + mutation * (rand() % 2 == 1 ? 1 : -1) * (rand() % (c_velocity[key_parent_1] == 0 ? 1 : c_velocity[key_parent_1]));
			next_p_angle[k] = p_angle[key_parent_2] + mutation * (rand() % 2 == 1 ? 1 : -1) * (rand() % (p_angle[key_parent_2] == 0 ? 1 : p_angle[key_parent_2]));
			next_p_velocity[k] = p_velocity[key_parent_2] + mutation * (rand() % 2 == 1 ? 1 : -1) * (rand() % (p_velocity[key_parent_2] == 0 ? 1 : p_velocity[key_parent_2]));
		}
		printf("\n");
	}
	printf("Solution:\n\tfitness = %d\n\tc1 = %d\n\tc2 = %d\n\tc3 = %d\n\tc4 = %d\n", fitness[best_key], c_position[best_key], c_velocity[best_key], p_angle[best_key], p_velocity[best_key]);
	terminateGPU();

	return 0; // comment to run tests


	#pragma mark -
	#pragma mark Debug
	printf("\nENTERING DEBUG SCOPE:\n\n");
	initiated = 0; // so the context is new

	#pragma mark - GPU test
	printf("GPU fitness again:\n");
	int k = generation_size;
	int * test_c_position = (int *) malloc(k * sizeof(int));
	int * test_c_velocity = (int *) malloc(k * sizeof(int));
	int * test_p_angle = (int *) malloc(k * sizeof(int));
	int * test_p_velocity = (int *) malloc(k * sizeof(int));
	int * test_fitness = (int *) malloc(k * sizeof(int));	
	for (int i = 0; i < k; i++) {
		test_c_position[i] = c_position[best_key];
		test_c_velocity[i] = c_velocity[best_key];
		test_p_angle[i] = p_angle[best_key];
		test_p_velocity[i] = p_velocity[best_key];
	}
	computeFitness(test_c_position, test_c_velocity, test_p_angle, test_p_velocity, test_fitness, 1);
	for (int i = 0; i < k; i++) {
		printf("Test Solution:\n\tfitness = %d\n\tc1 = %d\n\tc2 = %d\n\tc3 = %d\n\tc4 = %d\n", test_fitness[i], test_c_position[i], test_c_velocity[i], test_p_angle[i], test_p_velocity[i]);
		break; // since all the results are the same
	}
	terminateGPU();

	#pragma mark - CPU test and Visualization
	printf("CPU fitness:\n");
	
	char command[254];
	FILE *fp;
	char output[254];

	// link this to to the Visualization binary
	sprintf(command, "/Volumes/Data/Projects/PoleBalanceGPU/Visualization/build/Debug/Visualization %d %d %d %d", c_position[best_key], c_velocity[best_key], p_angle[best_key], p_velocity[best_key]);	
	fp = popen(command, "r");
	if (fp == NULL) {
		printf("Failed to run command\n" );
		exit;
	}
	while (fgets(output, sizeof(output), fp) != NULL) {
		printf("\t%s", output);
	}
	int cpu_fitness = atoi(output);

	// this might fail from time to time since CPU and GPU round implementation differs
	assert(fitness[best_key] == test_fitness[0] && fitness[best_key] == cpu_fitness);
	
	return 0;
}
