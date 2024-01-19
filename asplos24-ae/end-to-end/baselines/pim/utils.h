#pragma once
#include <sys/time.h>
#include <stdint.h>


struct transformer_time
{
	double amm_latency = 0.;
	double non_amm_latency = 0.;
};

struct amm_time
{
	double index_calc_latency = 0.;
	double data_transfer_latency = 0.;
	double kernel_latency = 0.;
	double other_latency = 0.;
};

struct transformer_amm_time
{
	double qkv_projection_latency = 0.;
	double o_projection_latency = 0.;
	double ffn1_latency = 0.;
	double ffn2_latency = 0.;
};

struct cnn_amm_time
{
	double conv0_latency = 0.;
	double conv1_latency = 0.;
	double conv2_latency = 0.;
	double conv3_latency = 0.;
	double residual_conv_latency = 0.;
};

static transformer_time transformer_profiles;
static amm_time amm_profiles;
static transformer_amm_time transformer_amm_profiles;
static cnn_amm_time cnn_amm_profiles;

static double W_time() {
  	timeval marker;
  	gettimeofday(&marker, NULL);
  	return ((double)(marker.tv_sec) * 1e6 + (double)(marker.tv_usec)) * 1e-6;
}
