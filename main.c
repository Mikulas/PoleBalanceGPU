/**
 * must click on "Episode 3" under the "Executables" list item, then get info (or apple-I) then set the working directory to Project Directory (not build directory)
 */


#include <stdio.h>
#include <assert.h>
#include <sys/sysctl.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <OpenCL/OpenCL.h>

#pragma mark -
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

#pragma mark -
#pragma mark Main OpenCL Routine
int computeFitness(int * c_position, int * c_velocity, int * p_angle, int * p_velocity, float * fitness, int n)
{
	cl_program program[1];
	cl_kernel kernel[2];
	
	cl_command_queue cmd_queue;
	cl_context   context;
	
	cl_device_id cpu = NULL, device = NULL;

	cl_int err = 0;
	size_t returned_size = 0;
	size_t buffer_size;
	
	cl_mem mem_c_position, mem_c_velocity, mem_p_angle, mem_p_velocity, mem_fitness;
	
#pragma mark Device Information
	{
		// Find the CPU CL device, as a fallback
		err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_CPU, 1, &cpu, NULL);
		assert(err == CL_SUCCESS);
		
		// Find the GPU CL device, this is what we really want
		// If there is no GPU device is CL capable, fall back to CPU
		err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
		if (err != CL_SUCCESS) device = cpu;
		assert(device);
	
		// Get some information about the returned device
		cl_char vendor_name[1024] = {0};
		cl_char device_name[1024] = {0};
		err = clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(vendor_name), vendor_name, &returned_size);
		err |= clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, &returned_size);
		assert(err == CL_SUCCESS);
		printf("Connecting to %s %s...\n", vendor_name, device_name);
	}
	
#pragma mark Context and Command Queue
	{
		// Now create a context to perform our calculation with the 
		// specified device 
		context = clCreateContext(0, 1, &device, NULL, NULL, &err);
		assert(err == CL_SUCCESS);
		
		// And also a command queue for the context
		cmd_queue = clCreateCommandQueue(context, device, 0, NULL);
	}
	
#pragma mark Program and Kernel Creation
	{
		// Load the program source from disk
		// The kernel/program is the project directory and in Xcode the executable
		// is set to launch from that directory hence we use a relative path
		const char * filename = "kernel.cl";
		char *program_source = load_program_source(filename);
		program[0] = clCreateProgramWithSource(context, 1, (const char**)&program_source, NULL, &err);
		assert(err == CL_SUCCESS);
		
		err = clBuildProgram(program[0], 0, NULL, NULL, NULL, NULL);
		assert(err == CL_SUCCESS);
		
		// Now create the kernel "objects" that we want to use in the example file 
		kernel[0] = clCreateKernel(program[0], "add", &err);
	}
		
#pragma mark Memory Allocation
	{
		// Allocate memory on the device to hold our data and store the results into
		buffer_size = sizeof(int) * n;
		
		mem_c_position = clCreateBuffer(context, CL_MEM_READ_ONLY, buffer_size, NULL, NULL);
		err = clEnqueueWriteBuffer(cmd_queue, mem_c_position, CL_TRUE, 0, buffer_size, (void*)c_position, 0, NULL, NULL);
		mem_c_velocity = clCreateBuffer(context, CL_MEM_READ_ONLY, buffer_size, NULL, NULL);
		err |= clEnqueueWriteBuffer(cmd_queue, mem_c_velocity, CL_TRUE, 0, buffer_size, (void*)c_velocity, 0, NULL, NULL);
		mem_p_angle = clCreateBuffer(context, CL_MEM_READ_ONLY, buffer_size, NULL, NULL);
		err |= clEnqueueWriteBuffer(cmd_queue, mem_p_angle, CL_TRUE, 0, buffer_size, (void*)p_angle, 0, NULL, NULL);
		mem_p_velocity = clCreateBuffer(context, CL_MEM_READ_ONLY, buffer_size, NULL, NULL);
		err |= clEnqueueWriteBuffer(cmd_queue, mem_p_velocity, CL_TRUE, 0, buffer_size, (void*)p_velocity, 0, NULL, NULL);
		assert(err == CL_SUCCESS);
		
		// Results array
		buffer_size = sizeof(float) * n;
		mem_fitness	= clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
		
		// Get all of the stuff written and allocated
		clFinish(cmd_queue);
	}
	
#pragma mark Kernel Arguments
	{
		// Now setup the arguments to our kernel
		err  = clSetKernelArg(kernel[0],  0, sizeof(cl_mem), &mem_c_position);
		err |= clSetKernelArg(kernel[0],  1, sizeof(cl_mem), &mem_c_velocity);
		err |= clSetKernelArg(kernel[0],  2, sizeof(cl_mem), &mem_p_angle);
		err |= clSetKernelArg(kernel[0],  3, sizeof(cl_mem), &mem_p_velocity);
		err |= clSetKernelArg(kernel[0],  4, sizeof(cl_mem), &mem_fitness);
		assert(err == CL_SUCCESS);
	}
	
#pragma mark Execution and Read
	{
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
	}
	
#pragma mark Teardown
	{
		clReleaseMemObject(mem_c_position);
		clReleaseMemObject(mem_c_velocity);
		clReleaseMemObject(mem_p_angle);
		clReleaseMemObject(mem_p_velocity);
		clReleaseMemObject(mem_fitness);
		
		clReleaseCommandQueue(cmd_queue);
		clReleaseContext(context);
	}
	return CL_SUCCESS;
}



int main (int argc, const char * argv[]) {
    
	// Problem size
	int generation_size = 500;
	int generation_count = 20;
	
	srand(time(NULL));
	
	int n;
	
	// Allocate some memory and a place for the results
	int * c_position = (int *) malloc(generation_size * sizeof(int));
	int * c_velocity = (int *) malloc(generation_size * sizeof(int));
	int * p_angle = (int *) malloc(generation_size * sizeof(int));
	int * p_velocity = (int *) malloc(generation_size * sizeof(int));
	float * fitness = (float *) malloc(generation_size * sizeof(float));
	float fitness_sum = 0;
	int best_key = 0;

	int * next_c_position = (int *) malloc(generation_size * sizeof(int));
	int * next_c_velocity = (int *) malloc(generation_size * sizeof(int));
	int * next_p_angle = (int *) malloc(generation_size * sizeof(int));
	int * next_p_velocity = (int *) malloc(generation_size * sizeof(int));

	// Generate first generation
	for (int i = 0; i < generation_size; i++) {
		next_c_position[i] = (rand() % 2 == 1 ? 1 : -1) * rand() % 100;
		next_c_velocity[i] = (rand() % 2 == 1 ? 1 : -1) * rand() % 100;
		next_p_angle[i] = (rand() % 2 == 1 ? 1 : -1) * rand() % 100;
		next_p_velocity[i] = (rand() % 2 == 1 ? 1 : -1) * rand() % 100;
		//fitness[i] = 0.f;
	}

	for (n = 0; n < generation_count; n++) {
		c_position = next_c_position;
		c_velocity = next_c_velocity;
		p_angle = next_p_angle;
		p_velocity = next_p_velocity;

		fitness_sum = 0;
		best_key = 0;

		// Do the OpenCL calculation
		computeFitness(c_position, c_velocity, p_angle, p_velocity, fitness, generation_size);

		int fitness_max = 0;
		float * border = (float *) malloc(generation_size * sizeof(float));
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

		for (int k = 0; k < generation_size; k++) {
			int key_parent_1 = 0;
			int key_parent_2 = 0;

			// get weighted entity (roulette wheel implementation)
			float roll = (float)(rand() % (int) ceil(100 * fitness_sum)) / 100;
			int i;
			for (i = 0; i < generation_size; i++) {
				if (roll < border[i]) {
					break;
				}
			}
			key_parent_1 = i;
			
			roll = (float)(rand() % (int) ceil(100 * fitness_sum)) / 100;
			for (int i = 0; i < generation_size; i++) {
				if (roll < border[i]) {
					break;
				}
			}
			key_parent_2 = i;

			next_c_position[k] = c_position[key_parent_1];
			next_c_velocity[k] = c_velocity[key_parent_1];
			next_p_angle[k] = p_angle[key_parent_2];
			next_p_velocity[k] = p_velocity[key_parent_2];
		}
	}

	printf("Solution:\n\tfitness: %f\n\tc1: %i\n\tc2: %i\n\tc3: %i\n\tc4: %i\n", fitness[best_key], c_position[best_key], c_velocity[best_key], p_angle[best_key], p_velocity[best_key]);

	return 0;
}








