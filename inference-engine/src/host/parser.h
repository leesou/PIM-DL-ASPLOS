#pragma once

#include <string>
#include <iostream>
#include <fstream>
#include "defines.h"
#include "yaml-cpp/yaml.h"


void parse_amm_configs(AMMParams& amm_params, std::string yaml_path);


void parse_transformer_configs(TransformerParams& transformer_params, std::string yaml_path);
