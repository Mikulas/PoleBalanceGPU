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
cl_program program;
cl_kernel kernel[3];
cl_command_queue cmd_queue;
cl_context context;
cl_device_id cpu = NULL, device = NULL;
cl_int err = 0;
size_t returned_size = 0;
size_t buffer_size;
cl_mem mem_scale, mem_c_position, mem_c_velocity, mem_p_angle, mem_p_velocity, mem_fitness;

#pragma mark Configuration
const int generation_size = 100;
const int generation_count = 300;
const float mutation = 0.3;
const int time_total = 20000; // should be the same as in kernel.cl
const int tournament_size = 10;

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
	program = clCreateProgramWithSource(context, 1, (const char**)&program_source, NULL, &err);
	assert(err == CL_SUCCESS);

	err |= clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	assert(err == CL_SUCCESS);

	// Now create the kernel "objects" that we want to use in the example file 
	kernel[0] = clCreateKernel(program, "computeFitness", &err);
	kernel[1] = clCreateKernel(program, "prepareScale", &err);
	kernel[2] = clCreateKernel(program, "nextGeneration", &err);
	assert(err == CL_SUCCESS);

	#pragma mark Memory Allocation
	// Allocate memory on the device to hold our data and store the results into
	buffer_size = sizeof(int) * n;

	mem_scale = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, &err);
	mem_c_position = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, &err);
	mem_c_velocity = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, &err);
	mem_p_angle = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, &err);
	mem_p_velocity = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, &err);
	mem_fitness = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, &err);
	assert(err == CL_SUCCESS);

	// Get all of the stuff written and allocated
	clFinish(cmd_queue);

	printf(" done\n");

	return err; // CL_SUCCESS
}

void terminateGPU()
{
	#pragma mark Teardown
	for (int i = 0; i < sizeof(kernel) / sizeof(cl_kernel); ++i) {
		clReleaseKernel(kernel[i]);
	}
	clReleaseProgram(program);
	clReleaseCommandQueue(cmd_queue);
	clReleaseContext(context);
	clReleaseMemObject(mem_scale);
	clReleaseMemObject(mem_c_position);
	clReleaseMemObject(mem_c_velocity);
	clReleaseMemObject(mem_p_angle);
	clReleaseMemObject(mem_p_velocity);
	clReleaseMemObject(mem_fitness);
}

void writeBuffer(int * c_position, int * c_velocity, int * p_angle, int * p_velocity, int * fitness, int n)
{
	#pragma mark Writing memory
	buffer_size = sizeof(int) * n;
	err = clEnqueueWriteBuffer(cmd_queue, mem_c_position, CL_FALSE, 0, buffer_size, (void *) c_position, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(cmd_queue, mem_c_velocity, CL_FALSE, 0, buffer_size, (void *) c_velocity, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(cmd_queue, mem_p_angle, CL_FALSE, 0, buffer_size, (void *) p_angle, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(cmd_queue, mem_p_velocity, CL_FALSE, 0, buffer_size, (void *) p_velocity, 0, NULL, NULL);
	assert(err == CL_SUCCESS);
	
	// Get all of the stuff written
	clFinish(cmd_queue);	
}

#pragma mark -
#pragma mark Generation context
int computeFitness(int n)
{
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
	return CL_SUCCESS;
}

int prepareScale()
{
	#pragma mark Kernel Arguments
	err  = clSetKernelArg(kernel[1], 0, sizeof(cl_mem), (void *) &mem_fitness);
	err |= clSetKernelArg(kernel[1], 1, sizeof(cl_mem), (void *) &mem_scale);
	assert(err == CL_SUCCESS);
	
	#pragma mark Execution and Reading memory
	size_t global_work_size = 1;
	err = clEnqueueNDRangeKernel(cmd_queue, kernel[1], 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
	assert(err == CL_SUCCESS);
	clFinish(cmd_queue);
	return CL_SUCCESS;
}

int nextGeneration(int n)
{
	#pragma mark Kernel Arguments
	err  = clSetKernelArg(kernel[2], 0, sizeof(cl_mem), (void *) &mem_c_position);
	err |= clSetKernelArg(kernel[2], 1, sizeof(cl_mem), (void *) &mem_c_velocity);
	err |= clSetKernelArg(kernel[2], 2, sizeof(cl_mem), (void *) &mem_p_angle);
	err |= clSetKernelArg(kernel[2], 3, sizeof(cl_mem), (void *) &mem_p_velocity);
	err |= clSetKernelArg(kernel[2], 4, sizeof(cl_mem), (void *) &mem_fitness);
	//err |= clSetKernelArg(kernel[2], 5, sizeof(cl_mem), (void *) &mem_scale);
	assert(err == CL_SUCCESS);
	
	#pragma mark Execution and Reading memory
	size_t global_work_size = n;
	err = clEnqueueNDRangeKernel(cmd_queue, kernel[2], 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
	assert(err == CL_SUCCESS);
	clFinish(cmd_queue);
	return CL_SUCCESS;
}

int readBuffer(int * c_position, int * c_velocity, int * p_angle, int * p_velocity, int * fitness, int n)
{
	// Once finished read back the results from the answer 
	// array into the results array
	buffer_size = sizeof(int) * n;
	err = clEnqueueReadBuffer(cmd_queue, mem_c_position, CL_FALSE, 0, buffer_size, c_position, 0, NULL, NULL);
	err = clEnqueueReadBuffer(cmd_queue, mem_c_velocity, CL_FALSE, 0, buffer_size, c_velocity, 0, NULL, NULL);
	err = clEnqueueReadBuffer(cmd_queue, mem_p_angle, CL_FALSE, 0, buffer_size, p_angle, 0, NULL, NULL);
	err = clEnqueueReadBuffer(cmd_queue, mem_p_velocity, CL_FALSE, 0, buffer_size, p_velocity, 0, NULL, NULL);
	err = clEnqueueReadBuffer(cmd_queue, mem_fitness, CL_FALSE, 0, buffer_size, fitness, 0, NULL, NULL);
	assert(err == CL_SUCCESS);
	clFinish(cmd_queue);
	return CL_SUCCESS;
}

#pragma mark -
int main (int argc, const char * argv[])
{
	srand(time(NULL));

	#pragma mark Allocate standard memory
	int * c_position = (int *) malloc(generation_size * sizeof(int));
	int * c_velocity = (int *) malloc(generation_size * sizeof(int));
	int * p_angle = (int *) malloc(generation_size * sizeof(int));
	int * p_velocity = (int *) malloc(generation_size * sizeof(int));
	int * fitness = (int *) malloc(generation_size * sizeof(int));
	
	#pragma mark Generate first generation
	for (int i = 0; i < generation_size; i++) {
		const int sign = rand() % 2 == 1 ? 1 : -1;
		c_position[i] = sign * rand() % 1000;
		c_velocity[i] = sign * rand() % 1000;
		p_angle[i] = sign * rand() % 1000;
		p_velocity[i] = sign * rand() % 1000;
		fitness[i] = 0;
	}
	
	initGPU(generation_size);
	writeBuffer(c_position, c_velocity, p_angle, p_velocity, fitness, generation_size);

	#pragma mark Genetical algorithm
	for (int n = 0; n < generation_count; n++) {
		printf("generation %d\n", n, generation_size);
		computeFitness(generation_size);
		prepareScale();
		nextGeneration(generation_size);
	}
	readBuffer(c_position, c_velocity, p_angle, p_velocity, fitness, generation_size);
	
	int best_key = 0;
	for (int n = 0; n < generation_size; n++) {
		if (fitness[n] > fitness[best_key])
			best_key = n;
	}
	printf("Solution:\n\tfitness = %d\n\tc1 = %d\n\tc2 = %d\n\tc3 = %d\n\tc4 = %d\n", fitness[best_key], c_position[best_key], c_velocity[best_key], p_angle[best_key], p_velocity[best_key]);
	terminateGPU();
	return 0;
}
