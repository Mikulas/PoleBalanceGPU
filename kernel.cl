/**
 * @author Mikuláš Dítě
 * @license Original BSD, see license.txt
 */

typedef struct {
	int x, y, z, w;
} Random;

/**
 * ND
 * READ-WRITE
 */
__kernel void computeFitness(__global int *c_c_position, __global int *c_c_velocity, __global int *c_p_angle, __global int *c_p_velocity, __global int *fitness)
{
	const float PI = 3.14159265;
	const int gid = get_global_id(0);
	const float fail_position = 2.4; // [meters]
	const float fail_angle = PI / 6; // [radians] 30 degrees
	const float g_acceleration = -9.81; // [meters/second/second]
	const float abs_force = 1; // [newtons]
	const float p_length = 0.5; // [meters] relative from pivot
	const float p_mass = 0.1; // [kilograms]
	const float c_mass = 1; // [kilograms]
	const int time_total = 20000; // [miliseconds]
	const int time_step = 25; // [miliseconds]

	// default values
	float c_position = 0;
	float c_velocity = 0;
	float c_acceleration = 0;
	float p_angle = PI / 36; // 5 degrees
	float p_velocity = 0;
	float p_acceleration = 0;
	float force; // this is set in each step to ±abs_force

	float delta = ((float) time_step / (float) 1000);

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


/**
 * ONE DIMENSIONAL
 * READ-WRITE
 */
__kernel void prepareScale(__global int *fitness, __global int *scale)
{
	const int dimension = 100; // MUST ALSO BE CHANGED IN generation_size IN CPU FILE!

	scale[0] = fitness[0];
	for (int i = 1; i < dimension; i++) {
		scale[i] = scale[i - 1] + fitness[i] + 1;
	}
}


/**
 * ND
 * READ-WRITE
 */
__kernel void nextGeneration(__global int *c_c_position, __global int *c_c_velocity, __global int *c_p_angle, __global int *c_p_velocity, __global int *fitness, __global int *scale, __global Random *random)
{
	const int dimension = get_global_size(0);
	const int gid = get_global_id(0);
	
	const int scale_max = scale[dimension - 1];
	
	int r1 = random[gid].w % scale_max;
	int r2 = random[gid].y % scale_max;
	int parent1 = dimension - 1;
	int parent2 = dimension - 1;
	
	for (int i = 0; i < dimension; i++) {
		if (r1 < scale[i])
			parent1 = i;
		if (r2 < scale[i])
			parent2 = i;
	}
		
	const int temp_c_position = c_c_position[parent2];
	const int temp_c_velocity = c_c_velocity[parent2];
	const int temp_p_angle = c_p_angle[parent2];
	const int temp_p_velocity = c_p_velocity[parent2];
	
	// switch two values between parents
	const float mutation = 0.05;
	const int sign = (random[gid].x - random[gid].y) > 0 ? 1 : -1;
	c_c_position[parent2] = c_c_position[parent1] + sign * mutation * (float)(random[gid].x % c_c_position[parent1]);
	c_c_velocity[parent2] = c_c_velocity[parent1] + sign * mutation * (float)(random[gid].x % c_c_position[parent1]);
	//c_p_angle[parent2] = c_p_angle[parent1];
	//c_p_velocity[parent2] = c_p_velocity[parent1];
		
	c_c_position[parent1] = temp_c_position + mutation * (float)(random[gid].x % c_c_position[parent1]);
	c_c_velocity[parent1] = temp_c_velocity + mutation * (float)(random[gid].x % c_c_position[parent1]);
	//c_p_angle[parent1] = temp_p_angle;
	//c_p_velocity[parent1] = temp_p_velocity;
}

/**
 * ND
 * READ-WRITE
 * Xorshift pseudorandom
 * @see http://en.wikipedia.org/wiki/Xorshift
 */
__kernel void generateRand(__global Random *random)
{
	const int gid = get_global_id(0);
	int t = random[gid].x ^ (random[gid].x << 11);
	random[gid].x = random[gid].y;
	random[gid].y = random[gid].z;
	random[gid].z = random[gid].w;
	random[gid].w = random[gid].w ^ (random[gid].w >> 19) ^ (t ^ (t >> 8));
}




































