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
#include <math.h>

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

#pragma mark Configuration
const int generation_size = 100;
const int generation_count = 300;
const float mutation = 0.3;
const int time_total = 20000; // should be the same as in kernel.cl
const int tournament_size = 10;

#pragma mark -
#pragma mark CPU
void computeFitnessCPU(int *c_c_position, int *c_c_velocity, int *c_p_angle, int *c_p_velocity, int *fitness)
{
	const float fail_position = 2.4; // [meters]
	const float fail_angle = M_PI / 6; // [radians] 30 degrees
	const float g_acceleration = -9.81; // [meters/second/second]
	const float abs_force = 1; // [newtons]
	const float p_length = 0.5; // [meters] relative from pivot
	const float p_mass = 0.1; // [kilograms]
	const float c_mass = 1; // [kilograms]
	// const int time_total = 20000; // [miliseconds]
	const int time_step = 25; // [miliseconds]
	
	float delta = ((float) time_step / (float) 1000);
	
	for (int gid = 0; gid < generation_size; gid++) {
		/** default values */
		float c_position = 0;
		float c_velocity = 0;
		float c_acceleration = 0;
		float p_angle = M_PI / 36; // 5 degrees
		float p_velocity = 0;
		float p_acceleration = 0;
		float force; // this is set in each step to ±abs_force
		
		int t;
		for (t = 0; t < time_total; t += time_step) {
			// http://www.profjrwhite.com/system_dynamics/sdyn/s7/s7invp2/s7invp2.html (7.61, 7.62)
			c_position = c_position + delta * c_velocity;
			p_angle = p_angle + delta * p_velocity;

			force = abs_force * ((c_c_position[gid] * c_position + c_c_velocity[gid] * c_velocity + c_p_angle[gid] * p_angle + c_p_velocity[gid] * p_velocity) > 0 ? 1 : -1); // intentionally not signum, cart must always move
			c_acceleration = (force + p_mass * p_length * sin(p_angle) * p_velocity * p_velocity - p_mass * g_acceleration * cos(p_angle) * sin(p_angle)) / (c_mass + p_mass - p_mass * cos(p_angle) * cos(p_angle));
			p_acceleration = (force * cos(p_angle) - g_acceleration * (c_mass + p_mass) * sin(p_angle) + p_mass * p_length * cos(p_angle) * sin(p_angle) * p_velocity) / (p_mass * p_length * cos(p_angle) * cos(p_angle) - (c_mass + p_mass) * p_length);

			c_velocity = c_velocity + delta * c_acceleration;
			p_velocity = p_velocity - delta * p_acceleration;

			if (c_position * (c_position > 0 ? 1 : -1) >= fail_position || p_angle * (p_angle > 0 ? 1 : -1) > fail_angle) {
				break;
			}
		}
		fitness[gid] = t;
	}
}

#pragma mark -
#pragma mark GPU
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
	//printf("Connecting to %s %s...", vendor_name, device_name);

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

	//printf(" done\n");

	return err; // CL_SUCCESS
}

void terminateGPU()
{
	#pragma mark Teardown
	initiated = 0;

	clReleaseMemObject(mem_c_position);
	clReleaseMemObject(mem_c_velocity);
	clReleaseMemObject(mem_p_angle);
	clReleaseMemObject(mem_p_velocity);
	clReleaseMemObject(mem_fitness);

	clReleaseCommandQueue(cmd_queue);
	clReleaseContext(context);
}

int computeFitnessGPU(int * c_position, int * c_velocity, int * p_angle, int * p_velocity, int * fitness, int n)
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

// tournament selection
int getParentKey(int * fitness)
{
	int key = -1;
	int * arena = (int *) malloc(tournament_size * sizeof(int));

	for (int l = 0; l < tournament_size; l++) {
		arena[l] = rand() % generation_size;
	}

	for (int l = 0; l < tournament_size; l++) {
		if (key == -1 || fitness[ arena[l] ] > fitness[key])
			key = arena[l];
	}

	return key;
}

#pragma mark -
int main (int argc, const char * argv[])
{
	srand(time(NULL));

	#pragma mark Allocate standard memory
	for (int test_sample = 0; test_sample < 10000; test_sample++) {
		int time_cpu = 0;
		int time_gpu = 0;

		for (int isGPU = 1; isGPU >= 0; isGPU--) {
			if (isGPU)
				time_gpu = clock();
			else
				time_cpu = clock();

			int * c_position = (int *) malloc(generation_size * sizeof(int));
			int * c_velocity = (int *) malloc(generation_size * sizeof(int));
			int * p_angle = (int *) malloc(generation_size * sizeof(int));
			int * p_velocity = (int *) malloc(generation_size * sizeof(int));
			int * fitness = (int *) malloc(generation_size * sizeof(int));
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
			}

			#pragma mark Genetical algorithm
			int n;
			int fitness_sum = 0;
			for (n = 0; n < generation_count; n++) {
				c_position = next_c_position;
				c_velocity = next_c_velocity;
				p_angle = next_p_angle;
				p_velocity = next_p_velocity;
				
				best_key = 0;

				if (isGPU)
					computeFitnessGPU(c_position, c_velocity, p_angle, p_velocity, fitness, generation_size);
				else
					computeFitnessCPU(c_position, c_velocity, p_angle, p_velocity, fitness);

				// prevent computing generation in the last cycle
				if (n == generation_count - 1) break;
				
				for (int i = 0; i < generation_size; i++) {
					fitness_sum += fitness[i];
					if (fitness[i] > fitness[best_key]) {
						best_key = i;
					}
				}
				// break if best solution is already found
				if (fitness[best_key] >= time_total) break;
				
				//printf("gen[%d] best_fitness = \t%d\n", n, fitness[best_key]);
				// Elite - always copy the best one
				next_c_position[0] = c_position[best_key];
				next_c_velocity[0] = c_velocity[best_key];
				next_p_angle[0] = p_angle[best_key];
				next_p_velocity[0] = p_velocity[best_key];
				
				for (int k = 1; k < generation_size; k++) {			
					int key_parent_1 = getParentKey(fitness);
					int key_parent_2 = getParentKey(fitness);
					
					// Prepare next generation as combination of two parents, with mutation
					int sign = rand() % 2 ? 1 : -1;
					//printf("%d\t", c_position[k]);
					next_c_position[k] = c_position[key_parent_1] + mutation * sign * 10;
					next_c_velocity[k] = c_velocity[key_parent_1] + mutation * sign * 10;
					next_p_angle[k] = p_angle[key_parent_2] + mutation * sign * 10;
					next_p_velocity[k] = p_velocity[key_parent_2] + mutation * sign * 10;
				}
				//printf("generation %d  \tavg = %f\t best = %d\n", n, (double) fitness_sum / (double) generation_size, fitness[best_key]);
				fitness_sum = 0;
			}
			//printf("Solution:\n\tfitness = %d\n\tc1 = %d\n\tc2 = %d\n\tc3 = %d\n\tc4 = %d\n", fitness[best_key], c_position[best_key], c_velocity[best_key], p_angle[best_key], p_velocity[best_key]);
			if (isGPU) {
				terminateGPU();
				time_gpu = clock() - time_gpu;
			} else {
				time_cpu = clock() - time_cpu;
			}
		}
		printf("test sample[%d] | \tGPU time = %d\tCPU time = %d\t => %s\n", test_sample, time_gpu, time_cpu, time_gpu > time_cpu ? "[GPU WON]" : "[CPU WON]");
	}
	return 0;
}
