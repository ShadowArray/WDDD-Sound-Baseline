// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung, George L. Yuan,
// Ali Bakhoda, Andrew Turner, Ivan Sham
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice, this
// list of conditions and the following disclaimer in the documentation and/or
// other materials provided with the distribution.
// Neither the name of The University of British Columbia nor the names of its
// contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


#include "gpu-sim.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "zlib.h"


#include "shader.h"
#include "dram.h"
#include "mem_fetch.h"

#include <time.h>
#include "gpu-cache.h"
#include "gpu-misc.h"
#include "delayqueue.h"
#include "shader.h"
#include "icnt_wrapper.h"
#include "dram.h"
#include "addrdec.h"
#include "stat-tool.h"
#include "l2cache.h"

#include "../cuda-sim/ptx-stats.h"
#include "../statwrapper.h"
#include "../abstract_hardware_model.h"
#include "../debug.h"
#include "../gpgpusim_entrypoint.h"
#include "../cuda-sim/cuda-sim.h"
#include "../trace.h"
#include "mem_latency_stat.h"
#include "power_stat.h"
#include "visualizer.h"
#include "stats.h"

#ifdef GPGPUSIM_POWER_MODEL
#include "power_interface.h"
#else
class  gpgpu_sim_wrapper {};
#endif

#include <stdio.h>
#include <string.h>
#include <iostream>
#include <sstream>
#include <string>

#define MAX(a,b) (((a)>(b))?(a):(b))

//Added on 2017-01-22, Hongwen
FILE *Con_kernel_log;

enum concrete_scheduler shader_scheduler;
unsigned L1D_cache_size = 0;
std::string L1D_indexing_function;
unsigned L1D_config_mshr_entries = 0;
std::string L1D_allocate_policy;

void g_get_stdout_filename(char *filename);
bool g_interactive_debugger_enabled=false;

unsigned long long  gpu_sim_cycle = 0;
unsigned long long  gpu_tot_sim_cycle = 0;

// Zhen: for bottleneck stats
class bottleneck_stats *g_bottleneck_stats;
// Zhen: for concurrent kernels
char *g_con_kernel_mode_opt = NULL;
char *g_con_kernel_max_tbs_opt = NULL;
unsigned g_num_con_kernels;
unsigned g_con_kernel_mode = CON_KERNEL_MODE_MANUAL;
unsigned g_con_kernel_max_tbs[MAX_CON_KERNELS] = {0};
kernel_info_t *g_con_kernels[MAX_CON_KERNELS] = {NULL};
unsigned g_con_kernel_max_warp_id[MAX_CON_KERNELS] = {0};
    
//MDB
bool g_bypass_all_l1d = false;
bool g_enable_MDB = false;
bool g_MDB_global = false;
bool g_MDB_on_TB = false;
unsigned g_thread_per_shader;
unsigned g_warp_size;
unsigned g_max_warps_per_shader;
unsigned g_max_warp_id = 0;
unsigned g_max_TB_id = 0;
//MRPB bypass-on-stall
bool g_enable_MRPB_reorder = false;
bool g_enable_MRPB_bp_on_stall = false;
unsigned g_bp_limit_num = -1;

bool g_enable_runtime_stats = false;        
unsigned g_kid = 0;

// performance counter for stalls due to congestion.
unsigned int gpu_stall_dramfull = 0; 
unsigned int gpu_stall_icnt2sh = 0;

/* Clock Domains */

#define  CORE  0x01
#define  L2    0x02
#define  DRAM  0x04
#define  ICNT  0x08  


#define MEM_LATENCY_STAT_IMPL

extern gpgpu_sim* g_the_gpu;


#include "mem_latency_stat.h"

void power_config::reg_options(class OptionParser * opp)
{


	  option_parser_register(opp, "-gpuwattch_xml_file", OPT_CSTR,
			  	  	  	  	 &g_power_config_name,"GPUWattch XML file",
	                   "gpuwattch.xml");

	   option_parser_register(opp, "-power_simulation_enabled", OPT_BOOL,
	                          &g_power_simulation_enabled, "Turn on power simulator (1=On, 0=Off)",
	                          "0");

	   option_parser_register(opp, "-power_per_cycle_dump", OPT_BOOL,
	                          &g_power_per_cycle_dump, "Dump detailed power output each cycle",
	                          "0");

	   // Output Data Formats
	   option_parser_register(opp, "-power_trace_enabled", OPT_BOOL,
	                          &g_power_trace_enabled, "produce a file for the power trace (1=On, 0=Off)",
	                          "0");

	   option_parser_register(opp, "-power_trace_zlevel", OPT_INT32,
	                          &g_power_trace_zlevel, "Compression level of the power trace output log (0=no comp, 9=highest)",
	                          "6");

	   option_parser_register(opp, "-steady_power_levels_enabled", OPT_BOOL,
	                          &g_steady_power_levels_enabled, "produce a file for the steady power levels (1=On, 0=Off)",
	                          "0");

	   option_parser_register(opp, "-steady_state_definition", OPT_CSTR,
			   	  &gpu_steady_state_definition, "allowed deviation:number of samples",
	                 	  "8:4");

}

void memory_config::reg_options(class OptionParser * opp)
{
    option_parser_register(opp, "-gpgpu_dram_scheduler", OPT_INT32, &scheduler_type, 
                                "0 = fifo, 1 = FR-FCFS (defaul)", "1");
    option_parser_register(opp, "-gpgpu_dram_partition_queues", OPT_CSTR, &gpgpu_L2_queue_config, 
                           "i2$:$2d:d2$:$2i",
                           "8:8:8:8");

    option_parser_register(opp, "-l2_ideal", OPT_BOOL, &l2_ideal, 
                           "Use a ideal L2 cache that always hit",
                           "0");
    option_parser_register(opp, "-gpgpu_cache:dl2", OPT_CSTR, &m_L2_config.m_config_string, 
                   "unified banked L2 data cache config "
                   " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<merge>,<mq>}",
                   "64:128:8,L:B:m:N,A:16:4,4");
    option_parser_register(opp, "-gpgpu_cache:dl2_texture_only", OPT_BOOL, &m_L2_texure_only, 
                           "L2 cache used for texture only",
                           "1");
    option_parser_register(opp, "-gpgpu_n_mem", OPT_UINT32, &m_n_mem, 
                 "number of memory modules (e.g. memory controllers) in gpu",
                 "8");
    option_parser_register(opp, "-gpgpu_n_sub_partition_per_mchannel", OPT_UINT32, &m_n_sub_partition_per_memory_channel, 
                 "number of memory subpartition in each memory module",
                 "1");
    option_parser_register(opp, "-gpgpu_n_mem_per_ctrlr", OPT_UINT32, &gpu_n_mem_per_ctrlr, 
                 "number of memory chips per memory controller",
                 "1");
    option_parser_register(opp, "-gpgpu_memlatency_stat", OPT_INT32, &gpgpu_memlatency_stat, 
                "track and display latency statistics 0x2 enables MC, 0x4 enables queue logs",
                "0");
    option_parser_register(opp, "-gpgpu_frfcfs_dram_sched_queue_size", OPT_INT32, &gpgpu_frfcfs_dram_sched_queue_size, 
                "0 = unlimited (default); # entries per chip",
                "0");
    option_parser_register(opp, "-gpgpu_dram_return_queue_size", OPT_INT32, &gpgpu_dram_return_queue_size, 
                "0 = unlimited (default); # entries per chip",
                "0");
    option_parser_register(opp, "-gpgpu_dram_buswidth", OPT_UINT32, &busW, 
                 "default = 4 bytes (8 bytes per cycle at DDR)",
                 "4");
    option_parser_register(opp, "-gpgpu_dram_burst_length", OPT_UINT32, &BL, 
                 "Burst length of each DRAM request (default = 4 data bus cycle)",
                 "4");
    option_parser_register(opp, "-dram_data_command_freq_ratio", OPT_UINT32, &data_command_freq_ratio, 
                 "Frequency ratio between DRAM data bus and command bus (default = 2 times, i.e. DDR)",
                 "2");
    option_parser_register(opp, "-gpgpu_dram_timing_opt", OPT_CSTR, &gpgpu_dram_timing_opt, 
                "DRAM timing parameters = {nbk:tCCD:tRRD:tRCD:tRAS:tRP:tRC:CL:WL:tCDLR:tWR:nbkgrp:tCCDL:tRTPL}",
                "4:2:8:12:21:13:34:9:4:5:13:1:0:0");
    option_parser_register(opp, "-rop_latency", OPT_UINT32, &rop_latency,
                     "ROP queue latency (default 85)",
                     "85");
    option_parser_register(opp, "-dram_latency", OPT_UINT32, &dram_latency,
                     "DRAM latency (default 30)",
                     "30");
    m_address_mapping.addrdec_setoption(opp);
    //option_parser_register(opp, "-icnt_flit_size", OPT_UINT32, &icnt_flit_size,
    //"Should be the same as interconnect configuration (default 32)",
    //"32");

    /*
    option_parser_register(opp, "-n_kernel1_TB", OPT_UINT32, &g_con_kernel_max_tbs[0], "(default 2)", "2");
    option_parser_register(opp, "-n_kernel2_TB", OPT_UINT32, &g_con_kernel_max_tbs[1], "(default 2)", "2");
    option_parser_register(opp, "-n_kernel3_TB", OPT_UINT32, &g_con_kernel_max_tbs[2], "(default 0)", "0");
    option_parser_register(opp, "-n_kernel4_TB", OPT_UINT32, &g_con_kernel_max_tbs[3], "(default 0)", "0");
    option_parser_register(opp, "-con_kernel_mode", OPT_UINT32, &g_con_kernel_mode, "(default 3 (manual))", "3");
    */
    option_parser_register(opp, "-num_con_kernels", OPT_UINT32, &g_num_con_kernels, "(default 2, useless if con_kernel is disabled)", "2");
    option_parser_register(opp, "-con_kernel_mode", OPT_CSTR, &g_con_kernel_mode_opt, "disable/even/full/manual", "manual");
    option_parser_register(opp, "-con_kernel_max_tbs", OPT_CSTR, &g_con_kernel_max_tbs_opt, "", "2:2");
    option_parser_register(opp, "-enable_runtime_stats", OPT_BOOL, &g_enable_runtime_stats, "", "0");

    //MDB
    option_parser_register(opp, "-bypass_all_l1d", OPT_BOOL, &g_bypass_all_l1d, "", "0");
    option_parser_register(opp, "-enable_MRPB_reorder", OPT_BOOL, &g_enable_MRPB_reorder, "", "0");
    option_parser_register(opp, "-enable_MRPB_bp_on_stall", OPT_BOOL, &g_enable_MRPB_bp_on_stall, "", "0");
    option_parser_register(opp, "-enable_MDB", OPT_BOOL, &g_enable_MDB, "", "0");
    option_parser_register(opp, "-MDB_global", OPT_BOOL, &g_MDB_global, "", "0");
    option_parser_register(opp, "-MDB_on_TB", OPT_BOOL, &g_MDB_on_TB, "", "0");
    option_parser_register(opp, "-bp_limit_num", OPT_UINT32, &g_bp_limit_num, "", "-1");

}

void shader_core_config::reg_options(class OptionParser * opp)
{
    option_parser_register(opp, "-gpgpu_simd_model", OPT_INT32, &model, 
                   "1 = post-dominator", "1");
    option_parser_register(opp, "-gpgpu_shader_core_pipeline", OPT_CSTR, &gpgpu_shader_core_pipeline_opt, 
                   "shader core pipeline config, i.e., {<nthread>:<warpsize>}",
                   "1024:32");
    option_parser_register(opp, "-gpgpu_tex_cache:l1", OPT_CSTR, &m_L1T_config.m_config_string, 
                   "per-shader L1 texture cache  (READ-ONLY) config "
                   " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<merge>,<mq>:<rf>}",
                   "8:128:5,L:R:m:N,F:128:4,128:2");
    option_parser_register(opp, "-gpgpu_const_cache:l1", OPT_CSTR, &m_L1C_config.m_config_string, 
                   "per-shader L1 constant memory cache  (READ-ONLY) config "
                   " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<merge>,<mq>} ",
                   "64:64:2,L:R:f:N,A:2:32,4" );
    option_parser_register(opp, "-gpgpu_cache:il1", OPT_CSTR, &m_L1I_config.m_config_string, 
                   "shader L1 instruction cache config "
                   " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<merge>,<mq>} ",
                   "4:256:4,L:R:f:N,A:2:32,4" );
    option_parser_register(opp, "-gpgpu_cache:dl1", OPT_CSTR, &m_L1D_config.m_config_string,
                   "per-shader L1 data cache config "
                   " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<merge>,<mq> | none}",
                   "none" );
    option_parser_register(opp, "-gpgpu_cache:dl1PrefL1", OPT_CSTR, &m_L1D_config.m_config_stringPrefL1,
                   "per-shader L1 data cache config "
                   " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<merge>,<mq> | none}",
                   "none" );
    option_parser_register(opp, "-gpgpu_cache:dl1PreShared", OPT_CSTR, &m_L1D_config.m_config_stringPrefShared,
                   "per-shader L1 data cache config "
                   " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<merge>,<mq> | none}",
                   "none" );
    option_parser_register(opp, "-gmem_skip_L1D", OPT_BOOL, &gmem_skip_L1D, 
                   "global memory access skip L1D cache (implements -Xptxas -dlcm=cg, default=no skip)",
                   "0");

    option_parser_register(opp, "-gpgpu_perfect_mem", OPT_BOOL, &gpgpu_perfect_mem, 
                 "enable perfect memory mode (no cache miss)",
                 "0");
    option_parser_register(opp, "-n_regfile_gating_group", OPT_UINT32, &n_regfile_gating_group,
                 "group of lanes that should be read/written together)",
                 "4");
    option_parser_register(opp, "-gpgpu_clock_gated_reg_file", OPT_BOOL, &gpgpu_clock_gated_reg_file,
                 "enable clock gated reg file for power calculations",
                 "0");
    option_parser_register(opp, "-gpgpu_clock_gated_lanes", OPT_BOOL, &gpgpu_clock_gated_lanes,
                 "enable clock gated lanes for power calculations",
                 "0");
    option_parser_register(opp, "-gpgpu_shader_registers", OPT_UINT32, &gpgpu_shader_registers, 
                 "Number of registers per shader core. Limits number of concurrent CTAs. (default 8192)",
                 "8192");
    option_parser_register(opp, "-gpgpu_shader_cta", OPT_UINT32, &max_cta_per_core, 
                 "Maximum number of concurrent CTAs in shader (default 8)",
                 "8");
    option_parser_register(opp, "-gpgpu_num_cta_barriers", OPT_UINT32, &max_barriers_per_cta,
                 "Maximum number of named barriers per CTA (default 16)",
                 "16");
    option_parser_register(opp, "-gpgpu_n_clusters", OPT_UINT32, &n_simt_clusters, 
                 "number of processing clusters",
                 "10");
    option_parser_register(opp, "-gpgpu_n_cores_per_cluster", OPT_UINT32, &n_simt_cores_per_cluster, 
                 "number of simd cores per cluster",
                 "3");
    option_parser_register(opp, "-gpgpu_n_cluster_ejection_buffer_size", OPT_UINT32, &n_simt_ejection_buffer_size, 
                 "number of packets in ejection buffer",
                 "8");
    option_parser_register(opp, "-gpgpu_n_ldst_response_buffer_size", OPT_UINT32, &ldst_unit_response_queue_size, 
                 "number of response packets in ld/st unit ejection buffer",
                 "2");
    option_parser_register(opp, "-gpgpu_shmem_size", OPT_UINT32, &gpgpu_shmem_size,
                 "Size of shared memory per shader core (default 16kB)",
                 "16384");
    /*
    option_parser_register(opp, "-gpgpu_shmem_size", OPT_UINT32, &gpgpu_shmem_sizeDefault,
                 "Size of shared memory per shader core (default 16kB)",
                 "16384");
    option_parser_register(opp, "-gpgpu_shmem_size_PrefL1", OPT_UINT32, &gpgpu_shmem_sizePrefL1,
                 "Size of shared memory per shader core (default 16kB)",
                 "16384");
    option_parser_register(opp, "-gpgpu_shmem_size_PrefShared", OPT_UINT32, &gpgpu_shmem_sizePrefShared,
                 "Size of shared memory per shader core (default 16kB)",
                 "16384");
    */
    
    option_parser_register(opp, "-gpgpu_shmem_num_banks", OPT_UINT32, &num_shmem_bank, 
                 "Number of banks in the shared memory in each shader core (default 16)",
                 "16");
    option_parser_register(opp, "-gpgpu_shmem_limited_broadcast", OPT_BOOL, &shmem_limited_broadcast, 
                 "Limit shared memory to do one broadcast per cycle (default on)",
                 "1");
    option_parser_register(opp, "-gpgpu_shmem_warp_parts", OPT_INT32, &mem_warp_parts,  
                 "Number of portions a warp is divided into for shared memory bank conflict check ",
                 "2");
    option_parser_register(opp, "-gpgpu_warpdistro_shader", OPT_INT32, &gpgpu_warpdistro_shader, 
                "Specify which shader core to collect the warp size distribution from", 
                "-1");
    option_parser_register(opp, "-gpgpu_warp_issue_shader", OPT_INT32, &gpgpu_warp_issue_shader, 
                "Specify which shader core to collect the warp issue distribution from", 
                "0");
    option_parser_register(opp, "-gpgpu_local_mem_map", OPT_BOOL, &gpgpu_local_mem_map, 
                "Mapping from local memory space address to simulated GPU physical address space (default = enabled)", 
                "1");
    option_parser_register(opp, "-gpgpu_num_reg_banks", OPT_INT32, &gpgpu_num_reg_banks, 
                "Number of register banks (default = 8)", 
                "8");
    option_parser_register(opp, "-gpgpu_reg_bank_use_warp_id", OPT_BOOL, &gpgpu_reg_bank_use_warp_id,
             "Use warp ID in mapping registers to banks (default = off)",
             "0");
    option_parser_register(opp, "-gpgpu_operand_collector_num_units_sp", OPT_INT32, &gpgpu_operand_collector_num_units_sp,
                "number of collector units (default = 4)", 
                "4");
    option_parser_register(opp, "-gpgpu_operand_collector_num_units_sfu", OPT_INT32, &gpgpu_operand_collector_num_units_sfu,
                "number of collector units (default = 4)", 
                "4");
    option_parser_register(opp, "-gpgpu_operand_collector_num_units_mem", OPT_INT32, &gpgpu_operand_collector_num_units_mem,
                "number of collector units (default = 2)", 
                "2");
    option_parser_register(opp, "-gpgpu_operand_collector_num_units_gen", OPT_INT32, &gpgpu_operand_collector_num_units_gen,
                "number of collector units (default = 0)", 
                "0");
    option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_sp", OPT_INT32, &gpgpu_operand_collector_num_in_ports_sp,
                           "number of collector unit in ports (default = 1)", 
                           "1");
    option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_sfu", OPT_INT32, &gpgpu_operand_collector_num_in_ports_sfu,
                           "number of collector unit in ports (default = 1)", 
                           "1");
    option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_mem", OPT_INT32, &gpgpu_operand_collector_num_in_ports_mem,
                           "number of collector unit in ports (default = 1)", 
                           "1");
    option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_gen", OPT_INT32, &gpgpu_operand_collector_num_in_ports_gen,
                           "number of collector unit in ports (default = 0)", 
                           "0");
    option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_sp", OPT_INT32, &gpgpu_operand_collector_num_out_ports_sp,
                           "number of collector unit in ports (default = 1)", 
                           "1");
    option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_sfu", OPT_INT32, &gpgpu_operand_collector_num_out_ports_sfu,
                           "number of collector unit in ports (default = 1)", 
                           "1");
    option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_mem", OPT_INT32, &gpgpu_operand_collector_num_out_ports_mem,
                           "number of collector unit in ports (default = 1)", 
                           "1");
    option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_gen", OPT_INT32, &gpgpu_operand_collector_num_out_ports_gen,
                           "number of collector unit in ports (default = 0)", 
                           "0");
    option_parser_register(opp, "-gpgpu_coalesce_arch", OPT_INT32, &gpgpu_coalesce_arch, 
                            "Coalescing arch (default = 13, anything else is off for now)", 
                            "13");
    option_parser_register(opp, "-gpgpu_num_sched_per_core", OPT_INT32, &gpgpu_num_sched_per_core, 
                            "Number of warp schedulers per core", 
                            "1");
    option_parser_register(opp, "-gpgpu_max_insn_issue_per_warp", OPT_INT32, &gpgpu_max_insn_issue_per_warp,
                            "Max number of instructions that can be issued per warp in one cycle by scheduler",
                            "2");
    option_parser_register(opp, "-gpgpu_simt_core_sim_order", OPT_INT32, &simt_core_sim_order,
                            "Select the simulation order of cores in a cluster (0=Fix, 1=Round-Robin)",
                            "1");
    option_parser_register(opp, "-gpgpu_pipeline_widths", OPT_CSTR, &pipeline_widths_string,
                            "Pipeline widths "
                            "ID_OC_SP,ID_OC_SFU,ID_OC_MEM,OC_EX_SP,OC_EX_SFU,OC_EX_MEM,EX_WB",
                            "1,1,1,1,1,1,1" );
    option_parser_register(opp, "-gpgpu_num_sp_units", OPT_INT32, &gpgpu_num_sp_units,
                            "Number of SP units (default=1)",
                            "1");
    option_parser_register(opp, "-gpgpu_num_sfu_units", OPT_INT32, &gpgpu_num_sfu_units,
                            "Number of SF units (default=1)",
                            "1");
    option_parser_register(opp, "-gpgpu_num_mem_units", OPT_INT32, &gpgpu_num_mem_units,
                            "Number if ldst units (default=1) WARNING: not hooked up to anything",
                             "1");
    option_parser_register(opp, "-gpgpu_scheduler", OPT_CSTR, &gpgpu_scheduler_string,
                                "Scheduler configuration: < lrr | gto | two_level_active > "
                                "If two_level_active:<num_active_warps>:<inner_prioritization>:<outer_prioritization>"
                                "For complete list of prioritization values see shader.h enum scheduler_prioritization_type"
                                "Default: gto",
                                 "gto");

    option_parser_register(opp, "-n_unthrottled_warps", OPT_INT32, &n_unthrottled_warps,
                            "number of unthrottled warps (default=1024)",
                            "1024");
}

void gpgpu_sim_config::reg_options(option_parser_t opp)
{
    gpgpu_functional_sim_config::reg_options(opp);
    m_shader_config.reg_options(opp);
    m_memory_config.reg_options(opp);
    power_config::reg_options(opp);
   option_parser_register(opp, "-gpgpu_max_cycle", OPT_INT32, &gpu_max_cycle_opt, 
               "terminates gpu simulation early (0 = no limit)",
               "0");
   option_parser_register(opp, "-gpgpu_max_insn", OPT_INT32, &gpu_max_insn_opt, 
               "terminates gpu simulation early (0 = no limit)",
               "0");
   option_parser_register(opp, "-gpgpu_max_cta", OPT_INT32, &gpu_max_cta_opt, 
               "terminates gpu simulation early (0 = no limit)",
               "0");
   option_parser_register(opp, "-gpgpu_runtime_stat", OPT_CSTR, &gpgpu_runtime_stat, 
                  "display runtime statistics such as dram utilization {<freq>:<flag>}",
                  "10000:0");
   option_parser_register(opp, "-liveness_message_freq", OPT_INT64, &liveness_message_freq, 
               "Minimum number of seconds between simulation liveness messages (0 = always print)",
               "1");
   option_parser_register(opp, "-gpgpu_flush_l1_cache", OPT_BOOL, &gpgpu_flush_l1_cache,
                "Flush L1 cache at the end of each kernel call",
                "0");
   option_parser_register(opp, "-gpgpu_flush_l2_cache", OPT_BOOL, &gpgpu_flush_l2_cache,
                   "Flush L2 cache at the end of each kernel call",
                   "0");

   option_parser_register(opp, "-gpgpu_deadlock_detect", OPT_BOOL, &gpu_deadlock_detect, 
                "Stop the simulation at deadlock (1=on (default), 0=off)", 
                "1");
   option_parser_register(opp, "-gpgpu_ptx_instruction_classification", OPT_INT32, 
               &gpgpu_ptx_instruction_classification, 
               "if enabled will classify ptx instruction types per kernel (Max 255 kernels now)", 
               "0");
   option_parser_register(opp, "-gpgpu_ptx_sim_mode", OPT_INT32, &g_ptx_sim_mode, 
               "Select between Performance (default) or Functional simulation (1)", 
               "0");
   option_parser_register(opp, "-gpgpu_clock_domains", OPT_CSTR, &gpgpu_clock_domains, 
                  "Clock Domain Frequencies in MhZ {<Core Clock>:<ICNT Clock>:<L2 Clock>:<DRAM Clock>}",
                  "500.0:2000.0:2000.0:2000.0");
   option_parser_register(opp, "-gpgpu_max_concurrent_kernel", OPT_INT32, &max_concurrent_kernel,
                          "maximum kernels that can run concurrently on GPU", "8" );
   option_parser_register(opp, "-gpgpu_cflog_interval", OPT_INT32, &gpgpu_cflog_interval, 
               "Interval between each snapshot in control flow logger", 
               "0");
   option_parser_register(opp, "-visualizer_enabled", OPT_BOOL,
                          &g_visualizer_enabled, "Turn on visualizer output (1=On, 0=Off)",
                          "1");
   option_parser_register(opp, "-visualizer_outputfile", OPT_CSTR, 
                          &g_visualizer_filename, "Specifies the output log file for visualizer",
                          NULL);
   option_parser_register(opp, "-visualizer_zlevel", OPT_INT32,
                          &g_visualizer_zlevel, "Compression level of the visualizer output log (0=no comp, 9=highest)",
                          "6");
    option_parser_register(opp, "-trace_enabled", OPT_BOOL, 
                          &Trace::enabled, "Turn on traces",
                          "0");
    option_parser_register(opp, "-trace_components", OPT_CSTR, 
                          &Trace::config_str, "comma seperated list of traces to enable. "
                          "Complete list found in trace_streams.tup. "
                          "Default none",
                          "none");
    option_parser_register(opp, "-trace_sampling_core", OPT_INT32, 
                          &Trace::sampling_core, "The core which is printed using CORE_DPRINTF. Default 0",
                          "0");
    option_parser_register(opp, "-trace_sampling_memory_partition", OPT_INT32, 
                          &Trace::sampling_memory_partition, "The memory partition which is printed using MEMPART_DPRINTF. Default -1 (i.e. all)",
                          "-1");

    option_parser_register(opp, "-bottleneck_stats_period", OPT_INT32, 
                          &bottleneck_stats_period, "Default 0, disabled",
                          "0");

    option_parser_register(opp, "-gpgpu_limit_cycles", OPT_INT64, 
                          &gpgpu_limit_cycles, "Default 0, disabled",
                          "0");
    option_parser_register(opp, "-gpgpu_limit_insns", OPT_INT64, 
                          &gpgpu_limit_insns, "Default 0, disabled",
                          "0");

    ptx_file_line_stats_options(opp);
   
}

/////////////////////////////////////////////////////////////////////////////

void increment_x_then_y_then_z( dim3 &i, const dim3 &bound)
{
   i.x++;
   if ( i.x >= bound.x ) {
      i.x = 0;
      i.y++;
      if ( i.y >= bound.y ) {
         i.y = 0;
         if( i.z < bound.z ) 
            i.z++;
      }
   }
}

void gpgpu_sim::launch( kernel_info_t *kinfo )
{
   unsigned cta_size = kinfo->threads_per_cta();
   if ( cta_size > m_shader_config->n_thread_per_shader ) {
      printf("Execution error: Shader kernel CTA (block) size is too large for microarch config.\n");
      printf("                 CTA size (x*y*z) = %u, max supported = %u\n", cta_size, 
             m_shader_config->n_thread_per_shader );
      printf("                 => either change -gpgpu_shader argument in gpgpusim.config file or\n");
      printf("                 modify the CUDA source to decrease the kernel block size.\n");
      abort();
   }
   unsigned n=0;
   for(n=0; n < m_running_kernels.size(); n++ ) {
       if( (NULL==m_running_kernels[n]) || m_running_kernels[n]->done() ) {
           m_running_kernels[n] = kinfo;
           break;
       }
   }
   assert(n < m_running_kernels.size());
}

bool gpgpu_sim::can_start_kernel()
{
   for(unsigned n=0; n < m_running_kernels.size(); n++ ) {
       if( (NULL==m_running_kernels[n]) || m_running_kernels[n]->done() ) 
           return true;
   }
   return false;
}

bool gpgpu_sim::get_more_cta_left() const
{ 
   if (m_config.gpu_max_cta_opt != 0) {
      if( m_total_cta_launched >= m_config.gpu_max_cta_opt )
          return false;
   }
   for(unsigned n=0; n < m_running_kernels.size(); n++ ) {
       if( m_running_kernels[n] && !m_running_kernels[n]->no_more_ctas_to_run() ) 
           return true;
   }
   return false;
}

kernel_info_t *gpgpu_sim::select_kernel()
{
  /*
  printf("gpgpu_sim::select_kernel: %u running_kernels\n", m_running_kernels.size());
  for (unsigned i = 0; i < m_running_kernels.size(); i++) {
    kernel_info_t *k = m_running_kernels[i];
    if (k) {
      printf("#%u: uid %u name %s\n", i, k->get_uid(), k->name().c_str());
    }
  }
  */
  
    for(unsigned n=0; n < m_running_kernels.size(); n++ ) {
        unsigned idx = (n+m_last_issued_kernel+1)%m_config.max_concurrent_kernel;
        if( m_running_kernels[idx] && !m_running_kernels[idx]->no_more_ctas_to_run() ) {
            m_last_issued_kernel=idx;
            // record this kernel for stat print if it is the first time this kernel is selected for execution  
            unsigned launch_uid = m_running_kernels[idx]->get_uid(); 
            if (std::find(m_executed_kernel_uids.begin(), m_executed_kernel_uids.end(), launch_uid) == m_executed_kernel_uids.end()) {
               m_executed_kernel_uids.push_back(launch_uid); 
               m_executed_kernel_names.push_back(m_running_kernels[idx]->name()); 
            }

            return m_running_kernels[idx];
        }
    }
    return NULL;
}

unsigned gpgpu_sim::finished_kernel()
{
    if( m_finished_kernel.empty() ) 
        return 0;
    unsigned result = m_finished_kernel.front();
    m_finished_kernel.pop_front();
    return result;
}

void gpgpu_sim::set_kernel_done( kernel_info_t *kernel ) 
{ 
    unsigned uid = kernel->get_uid();
    m_finished_kernel.push_back(uid);
    std::vector<kernel_info_t*>::iterator k;
    for( k=m_running_kernels.begin(); k!=m_running_kernels.end(); k++ ) {
        if( *k == kernel ) {
            *k = NULL;
            break;
        }
    }
    assert( k != m_running_kernels.end() ); 
}

void set_ptx_warp_size(const struct core_config * warp_size);

gpgpu_sim::gpgpu_sim( const gpgpu_sim_config &config ) 
    : gpgpu_t(config), m_config(config)
{ 
    m_shader_config = &m_config.m_shader_config;
    m_memory_config = &m_config.m_memory_config;
    set_ptx_warp_size(m_shader_config);
    ptx_file_line_stats_create_exposed_latency_tracker(m_config.num_shader());

#ifdef GPGPUSIM_POWER_MODEL
        m_gpgpusim_wrapper = new gpgpu_sim_wrapper(config.g_power_simulation_enabled,config.g_power_config_name);
#endif

    m_shader_stats = new shader_core_stats(m_shader_config);
    m_memory_stats = new memory_stats_t(m_config.num_shader(),m_shader_config,m_memory_config);
    average_pipeline_duty_cycle = (float *)malloc(sizeof(float));
    active_sms=(float *)malloc(sizeof(float));
    m_power_stats = new power_stat_t(m_shader_config,average_pipeline_duty_cycle,active_sms,m_shader_stats,m_memory_config,m_memory_stats);

    gpu_sim_insn = 0;
    gpu_tot_sim_insn = 0;
    gpu_tot_issued_cta = 0;
    gpu_deadlock = false;


    m_cluster = new simt_core_cluster*[m_shader_config->n_simt_clusters];
    for (unsigned i=0;i<m_shader_config->n_simt_clusters;i++) 
        m_cluster[i] = new simt_core_cluster(this,i,m_shader_config,m_memory_config,m_shader_stats,m_memory_stats);

    m_memory_partition_unit = new memory_partition_unit*[m_memory_config->m_n_mem];
    m_memory_sub_partition = new memory_sub_partition*[m_memory_config->m_n_mem_sub_partition];
    for (unsigned i=0;i<m_memory_config->m_n_mem;i++) {
        m_memory_partition_unit[i] = new memory_partition_unit(i, m_memory_config, m_memory_stats);
        for (unsigned p = 0; p < m_memory_config->m_n_sub_partition_per_memory_channel; p++) {
            unsigned submpid = i * m_memory_config->m_n_sub_partition_per_memory_channel + p; 
            m_memory_sub_partition[submpid] = m_memory_partition_unit[i]->get_sub_partition(p); 
        }
    }

    icnt_wrapper_init();
    icnt_create(m_shader_config->n_simt_clusters,m_memory_config->m_n_mem_sub_partition);

    time_vector_create(NUM_MEM_REQ_STAT);
    fprintf(stdout, "GPGPU-Sim uArch: performance model initialization complete.\n");

    m_running_kernels.resize( config.max_concurrent_kernel, NULL );
    m_last_issued_kernel = 0;
    m_last_cluster_issue = 0;
    *average_pipeline_duty_cycle=0;
    *active_sms=0;

    last_liveness_message_time = 0;

    g_bottleneck_stats = new bottleneck_stats();

    //Added on 2017-01-23, Hongwen
	if((Con_kernel_log = fopen("Con_kernel_log.txt","a"))==NULL){
		fprintf(Con_kernel_log, "Con_kernel_log.txt file open failed\n");
	    exit(0);
    }
}

int gpgpu_sim::shared_mem_size() const
{
   return m_shader_config->gpgpu_shmem_size;
}

int gpgpu_sim::num_registers_per_core() const
{
   return m_shader_config->gpgpu_shader_registers;
}

int gpgpu_sim::wrp_size() const
{
   return m_shader_config->warp_size;
}

int gpgpu_sim::shader_clock() const
{
   return m_config.core_freq/1000;
}

void gpgpu_sim::set_prop( cudaDeviceProp *prop )
{
   m_cuda_properties = prop;
}

const struct cudaDeviceProp *gpgpu_sim::get_prop() const
{
   return m_cuda_properties;
}

enum divergence_support_t gpgpu_sim::simd_model() const
{
   return m_shader_config->model;
}

void gpgpu_sim_config::init_clock_domains(void ) 
{
   sscanf(gpgpu_clock_domains,"%lf:%lf:%lf:%lf", 
          &core_freq, &icnt_freq, &l2_freq, &dram_freq);
   core_freq = core_freq MhZ;
   icnt_freq = icnt_freq MhZ;
   l2_freq = l2_freq MhZ;
   dram_freq = dram_freq MhZ;        
   core_period = 1/core_freq;
   icnt_period = 1/icnt_freq;
   dram_period = 1/dram_freq;
   l2_period = 1/l2_freq;
   printf("GPGPU-Sim uArch: clock freqs: %lf:%lf:%lf:%lf\n",core_freq,icnt_freq,l2_freq,dram_freq);
   printf("GPGPU-Sim uArch: clock periods: %.20lf:%.20lf:%.20lf:%.20lf\n",core_period,icnt_period,l2_period,dram_period);
}

void gpgpu_sim::reinit_clock_domains(void)
{
   core_time = 0;
   dram_time = 0;
   icnt_time = 0;
   l2_time = 0;
}

bool gpgpu_sim::active()
{
    if (m_config.gpu_max_cycle_opt && (gpu_tot_sim_cycle + gpu_sim_cycle) >= m_config.gpu_max_cycle_opt) 
       return false;
    if (m_config.gpu_max_insn_opt && (gpu_tot_sim_insn + gpu_sim_insn) >= m_config.gpu_max_insn_opt) 
       return false;
    if (m_config.gpu_max_cta_opt && (gpu_tot_issued_cta >= m_config.gpu_max_cta_opt) )
       return false;
    if (m_config.gpu_deadlock_detect && gpu_deadlock) 
       return false;
    for (unsigned i=0;i<m_shader_config->n_simt_clusters;i++) 
       if( m_cluster[i]->get_not_completed()>0 ) 
           return true;;
    for (unsigned i=0;i<m_memory_config->m_n_mem;i++) 
       if( m_memory_partition_unit[i]->busy()>0 )
           return true;;
    if( icnt_busy() )
        return true;
    if( get_more_cta_left() )
        return true;
    return false;
}

void gpgpu_sim::init()
{
    // run a CUDA grid on the GPU microarchitecture simulator
    gpu_sim_cycle = 0;
    gpu_sim_insn = 0;
    last_gpu_sim_insn = 0;
    m_total_cta_launched=0;

    reinit_clock_domains();
    set_param_gpgpu_num_shaders(m_config.num_shader());
    for (unsigned i=0;i<m_shader_config->n_simt_clusters;i++) 
       m_cluster[i]->reinit();
    m_shader_stats->new_grid();
    // initialize the control-flow, memory access, memory latency logger
    if (m_config.g_visualizer_enabled) {
        create_thread_CFlogger( m_config.num_shader(), m_shader_config->n_thread_per_shader, 0, m_config.gpgpu_cflog_interval );
    }
    shader_CTA_count_create( m_config.num_shader(), m_config.gpgpu_cflog_interval);
    if (m_config.gpgpu_cflog_interval != 0) {
       insn_warp_occ_create( m_config.num_shader(), m_shader_config->warp_size );
       shader_warp_occ_create( m_config.num_shader(), m_shader_config->warp_size, m_config.gpgpu_cflog_interval);
       shader_mem_acc_create( m_config.num_shader(), m_memory_config->m_n_mem, 4, m_config.gpgpu_cflog_interval);
       shader_mem_lat_create( m_config.num_shader(), m_config.gpgpu_cflog_interval);
       shader_cache_access_create( m_config.num_shader(), 3, m_config.gpgpu_cflog_interval);
       set_spill_interval (m_config.gpgpu_cflog_interval * 40);
    }

    if (g_network_mode)
       icnt_init();

    // McPAT initialization function. Called on first launch of GPU
#ifdef GPGPUSIM_POWER_MODEL
    if(m_config.g_power_simulation_enabled){
        init_mcpat(m_config, m_gpgpusim_wrapper, m_config.gpu_stat_sample_freq,  gpu_tot_sim_insn, gpu_sim_insn);
    }
#endif

    //Added on 2017-01-23, Hongwen
	for(unsigned i=0;i<4;i++){
	    gpu_sim_insn_kernel[i] = 0;
    }
}

void gpgpu_sim::update_stats() {
    m_memory_stats->memlatstat_lat_pw();
    gpu_tot_sim_cycle += gpu_sim_cycle;
    gpu_tot_sim_insn += gpu_sim_insn;
}

void gpgpu_sim::print_stats()
{
    ptx_file_line_stats_write_file();
    gpu_print_stat();

    if (g_network_mode) {
        printf("----------------------------Interconnect-DETAILS--------------------------------\n" );
        icnt_display_stats();
        icnt_display_overall_stats();
        printf("----------------------------END-of-Interconnect-DETAILS-------------------------\n" );
    }
}

void gpgpu_sim::deadlock_check()
{
   if (m_config.gpu_deadlock_detect && gpu_deadlock) {
      fflush(stdout);
      printf("\n\nGPGPU-Sim uArch: ERROR ** deadlock detected: last writeback core %u @ gpu_sim_cycle %u (+ gpu_tot_sim_cycle %u) (%u cycles ago)\n", 
             gpu_sim_insn_last_update_sid,
             (unsigned) gpu_sim_insn_last_update, (unsigned) (gpu_tot_sim_cycle-gpu_sim_cycle),
             (unsigned) (gpu_sim_cycle - gpu_sim_insn_last_update )); 
      unsigned num_cores=0;
      for (unsigned i=0;i<m_shader_config->n_simt_clusters;i++) {
         unsigned not_completed = m_cluster[i]->get_not_completed();
         if( not_completed ) {
             if ( !num_cores )  {
                 printf("GPGPU-Sim uArch: DEADLOCK  shader cores no longer committing instructions [core(# threads)]:\n" );
                 printf("GPGPU-Sim uArch: DEADLOCK  ");
                 m_cluster[i]->print_not_completed(stdout);
             } else if (num_cores < 8 ) {
                 m_cluster[i]->print_not_completed(stdout);
             } else if (num_cores >= 8 ) {
                 printf(" + others ... ");
             }
             num_cores+=m_shader_config->n_simt_cores_per_cluster;
         }
      }
      printf("\n");
      for (unsigned i=0;i<m_memory_config->m_n_mem;i++) {
         bool busy = m_memory_partition_unit[i]->busy();
         if( busy ) 
             printf("GPGPU-Sim uArch DEADLOCK:  memory partition %u busy\n", i );
      }
      if( icnt_busy() ) {
         printf("GPGPU-Sim uArch DEADLOCK:  iterconnect contains traffic\n");
         icnt_display_state( stdout );
      }
      printf("\nRe-run the simulator in gdb and use debug routines in .gdbinit to debug this\n");
      fflush(stdout);
      abort();
   }
}

/// printing the names and uids of a set of executed kernels (usually there is only one)
std::string gpgpu_sim::executed_kernel_info_string() 
{
   std::stringstream statout; 

   statout << "kernel_name = "; 
   for (unsigned int k = 0; k < m_executed_kernel_names.size(); k++) {
      statout << m_executed_kernel_names[k] << " "; 
   }
   statout << std::endl; 
   statout << "kernel_launch_uid = ";
   for (unsigned int k = 0; k < m_executed_kernel_uids.size(); k++) {
      statout << m_executed_kernel_uids[k] << " "; 
   }
   statout << std::endl; 

   return statout.str(); 
}
void gpgpu_sim::set_cache_config(std::string kernel_name,  FuncCache cacheConfig )
{
	m_special_cache_config[kernel_name]=cacheConfig ;
}

FuncCache gpgpu_sim::get_cache_config(std::string kernel_name)
{
	for (	std::map<std::string, FuncCache>::iterator iter = m_special_cache_config.begin(); iter != m_special_cache_config.end(); iter++){
		    std::string kernel= iter->first;
			if (kernel_name.compare(kernel) == 0){
				return iter->second;
			}
	}
	return (FuncCache)0;
}

bool gpgpu_sim::has_special_cache_config(std::string kernel_name)
{
	for (	std::map<std::string, FuncCache>::iterator iter = m_special_cache_config.begin(); iter != m_special_cache_config.end(); iter++){
	    	std::string kernel= iter->first;
			if (kernel_name.compare(kernel) == 0){
				return true;
			}
	}
	return false;
}


void gpgpu_sim::set_cache_config(std::string kernel_name)
{

	if(has_special_cache_config(kernel_name)){
		change_cache_config(get_cache_config(kernel_name));
	}else{
		change_cache_config(FuncCachePreferNone);
	}
}


void gpgpu_sim::change_cache_config(FuncCache cache_config)
{
  

	if(cache_config != m_shader_config->m_L1D_config.get_cache_status()){
		printf("FLUSH L1 Cache at configuration change between kernels\n");
		for (unsigned i=0;i<m_shader_config->n_simt_clusters;i++) {
			m_cluster[i]->cache_flush();
	    }
	}

	// Zhen: always use prefernone option
	m_shader_config->m_L1D_config.init(m_shader_config->m_L1D_config.m_config_string, FuncCachePreferNone);

	/*
	switch(cache_config){
	case FuncCachePreferNone:
		m_shader_config->m_L1D_config.init(m_shader_config->m_L1D_config.m_config_string, FuncCachePreferNone);
		m_shader_config->gpgpu_shmem_size=m_shader_config->gpgpu_shmem_sizeDefault;
		break;
	case FuncCachePreferL1:
		if((m_shader_config->m_L1D_config.m_config_stringPrefL1 == NULL) || (m_shader_config->gpgpu_shmem_sizePrefL1 == (unsigned)-1))
		{
			printf("WARNING: missing Preferred L1 configuration\n");
			m_shader_config->m_L1D_config.init(m_shader_config->m_L1D_config.m_config_string, FuncCachePreferNone);
			m_shader_config->gpgpu_shmem_size=m_shader_config->gpgpu_shmem_sizeDefault;

		}else{
			m_shader_config->m_L1D_config.init(m_shader_config->m_L1D_config.m_config_stringPrefL1, FuncCachePreferL1);
			m_shader_config->gpgpu_shmem_size=m_shader_config->gpgpu_shmem_sizePrefL1;
		}
		break;
	case FuncCachePreferShared:
		if((m_shader_config->m_L1D_config.m_config_stringPrefShared == NULL) || (m_shader_config->gpgpu_shmem_sizePrefShared == (unsigned)-1))
		{
			printf("WARNING: missing Preferred L1 configuration\n");
			m_shader_config->m_L1D_config.init(m_shader_config->m_L1D_config.m_config_string, FuncCachePreferNone);
			m_shader_config->gpgpu_shmem_size=m_shader_config->gpgpu_shmem_sizeDefault;
		}else{
			m_shader_config->m_L1D_config.init(m_shader_config->m_L1D_config.m_config_stringPrefShared, FuncCachePreferShared);
			m_shader_config->gpgpu_shmem_size=m_shader_config->gpgpu_shmem_sizePrefShared;
		}
		break;
	default:
		break;
	}
	*/
}


void gpgpu_sim::clear_executed_kernel_info()
{
   m_executed_kernel_names.clear();
   m_executed_kernel_uids.clear();
}

//Added on 2017-01-21, Hongwen
// Get current date/time, format is YYYY-MM-DD.HH:mm:ss
const std::string currentDateTime() {
  time_t     now = time(0);
		struct tm  tstruct;
		char       buf[80];
		tstruct = *localtime(&now);
		// Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
		// for more information about date/time format
		strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);

		return buf;
}

std::string gpgpu_sim::executed_kernel_info_string_con() 
{
   std::stringstream statout; 

   statout << "kernel_name = "; 
   for (unsigned int k = 0; k < m_executed_kernel_names.size(); k++) {
      statout << m_executed_kernel_names[k] << "----"; 
   }
   //statout << std::endl; 
   statout << "kernel_launch_uid = ";
   for (unsigned int k = 0; k < m_executed_kernel_uids.size(); k++) {
      statout << m_executed_kernel_uids[k] << "----"; 
   }
   //statout << std::endl; 

   return statout.str(); 
}

unsigned gpgpu_sim::concurrent_kernel_number()
{
    return m_executed_kernel_uids.size();    
}

void gpgpu_sim::gpu_print_stat() 
{  
   FILE *statfout = stdout; 

   std::string kernel_info_str = executed_kernel_info_string(); 
   fprintf(statfout, "%s", kernel_info_str.c_str()); 
   
    //Added on 2017-02-04, Hongwen
    /*
     char fname_icache[1024];
    g_get_stdout_filename(fname_icache);
    strcat(fname_icache, ".icache_stats.out");
    FILE *ofile_icache = fopen(fname_icache, "a");
    std::string con_kernel_info_str = executed_kernel_info_string_con(); 
    fprintf(ofile_icache, "%s\n", con_kernel_info_str.c_str());
    unsigned icache_line_size = m_shader_config->m_L1I_config.get_line_sz();
    unsigned icache_size = m_shader_config->m_L1I_config.get_line_sz() * m_shader_config->m_L1I_config.get_num_lines();
    g_bottleneck_stats->print_icache_stats(ofile_icache, icache_line_size, icache_size);
    fclose(ofile_icache);
    */

    //Added on 2017-02-24, Hongwen
    //print runtime statistics
    if(g_enable_runtime_stats){
        //Winst
        char fname_Winst[1024];
        g_get_stdout_filename(fname_Winst);
        strcat(fname_Winst, ".Winst_stats.out");
        FILE *fWinst_stats = fopen(fname_Winst, "a");
        g_bottleneck_stats->print_con_kernel_runtime_stats(fWinst_stats, g_bottleneck_stats->runtime_Winst_stats);
        fclose(fWinst_stats);
        //Cinst
        char fname_Cinst[1024];
        g_get_stdout_filename(fname_Cinst);
        strcat(fname_Cinst, ".Cinst_stats.out");
        FILE *fCinst_stats = fopen(fname_Cinst, "a");
        g_bottleneck_stats->print_con_kernel_runtime_stats(fCinst_stats, g_bottleneck_stats->runtime_Cinst_stats);
        fclose(fCinst_stats);
        //Minst
        char fname_Minst[1024];
        g_get_stdout_filename(fname_Minst);
        strcat(fname_Minst, ".Minst_stats.out");
        FILE *fMinst_stats = fopen(fname_Minst, "a");
        g_bottleneck_stats->print_con_kernel_runtime_stats(fMinst_stats, g_bottleneck_stats->runtime_Minst_stats);
        fclose(fMinst_stats);
        //l1d_access
        char fname_l1d_access[1024];
        g_get_stdout_filename(fname_l1d_access);
        strcat(fname_l1d_access, ".l1d_access_stats.out");
        FILE *fl1d_access_stats = fopen(fname_l1d_access, "a");
        g_bottleneck_stats->print_con_kernel_runtime_stats(fl1d_access_stats, g_bottleneck_stats->runtime_l1d_access_stats);
        fclose(fl1d_access_stats);
    }

   g_bottleneck_stats->print_con_kernels(stdout);
   char fname[1024];
   g_get_stdout_filename(fname);
   strcat(fname, ".bottleneck_stats.out");
   FILE *bn_stats = fopen(fname, "a");
   g_bottleneck_stats->print_con_kernels(bn_stats);
   fclose(bn_stats);
   //Added on 2017-01-21, Hongwen	
   {
        unsigned con_kernel_cnt = concurrent_kernel_number();
		
        std::stringstream caseout; 
		caseout << "result_" << con_kernel_cnt << "kernel_"; 
    
        caseout << "_L1D_" << L1D_cache_size << "KB";
		caseout << "_" << L1D_indexing_function; 
		caseout << "_" << L1D_config_mshr_entries << "MSHR";
		caseout << "_" << L1D_allocate_policy;
		//caseout << "_L2_MSHR_" << L2_config_mshr_entries;
		//caseout << "_ALLOC_" << L2_allocate_policy;
	    #ifdef RR_SCHEDULER
			caseout << "_RR_SCHEDULER";
        #else
			caseout << "_PRI_SCHEDULER";
        #endif
		if(shader_scheduler == CONCRETE_SCHEDULER_LRR)
			caseout << "_LRR";
		else if(shader_scheduler == CONCRETE_SCHEDULER_TWO_LEVEL_ACTIVE)
			caseout << "_TL";
		else if(shader_scheduler == CONCRETE_SCHEDULER_GTO)
			caseout << "_GTO";
    

        caseout << "_" << gpu_sim_cycle << "_cycles";
		
		float average_power = 0.0;
        #ifdef GPGPUSIM_POWER_MODEL
        if(m_config.g_power_simulation_enabled){
	        average_power = m_gpgpusim_wrapper->print_power_kernel_stats(gpu_sim_cycle, gpu_tot_sim_cycle, gpu_tot_sim_insn + gpu_sim_insn, kernel_info_str, true );
	        mcpat_reset_perf_count(m_gpgpusim_wrapper);
        }
        #endif
	
	//1. date-time, kernel_info, case, cycle, power 
	std::string kernel_info_str_con = executed_kernel_info_string_con(); 
	fprintf(Con_kernel_log, "%s,%s,%s,%d,%.4f",currentDateTime().c_str(), kernel_info_str_con.c_str(), caseout.str().c_str(), gpu_sim_cycle, average_power);
	//2. instruction counts: inst_0..3, tot
    unsigned gpu_sim_insn_kernel[MAX_CON_KERNELS];
    unsigned gpu_sim_insn_tot = 0;
    for (unsigned i = 0; i < MAX_CON_KERNELS; i++){
        gpu_sim_insn_kernel[i] = g_bottleneck_stats->thread_insn[i];
        gpu_sim_insn_tot += gpu_sim_insn_kernel[i];
    }
	fprintf(Con_kernel_log, ",    %d,%d,%d,%d,%d", gpu_sim_insn_kernel[0], gpu_sim_insn_kernel[1], gpu_sim_insn_kernel[2], gpu_sim_insn_kernel[3], gpu_sim_insn_tot);
	//3. ipc: ipc_0..3, tot
    float gpu_sim_ipc_kernel[MAX_CON_KERNELS];
    for (unsigned i = 0; i < MAX_CON_KERNELS; i++)
        gpu_sim_ipc_kernel[i] = (float)gpu_sim_insn_kernel[i] / gpu_sim_cycle;
    float gpu_sim_ipc_tot = (float)gpu_sim_insn_tot / gpu_sim_cycle;
	fprintf(Con_kernel_log, ",    %.4f,%.4f,%.4f,%.4f,%.4f", gpu_sim_ipc_kernel[0], gpu_sim_ipc_kernel[1],gpu_sim_ipc_kernel[2], gpu_sim_ipc_kernel[3], gpu_sim_ipc_tot);
	fprintf(Con_kernel_log, "\n");
   }

   printf("gpu_sim_cycle = %lld\n", gpu_sim_cycle);
   printf("gpu_sim_insn = %lld\n", gpu_sim_insn);
   printf("gpu_ipc = %12.4f\n", (float)gpu_sim_insn / gpu_sim_cycle);
   printf("gpu_tot_sim_cycle = %lld\n", gpu_tot_sim_cycle+gpu_sim_cycle);
   printf("gpu_tot_sim_insn = %lld\n", gpu_tot_sim_insn+gpu_sim_insn);
   printf("gpu_tot_ipc = %12.4f\n", (float)(gpu_tot_sim_insn+gpu_sim_insn) / (gpu_tot_sim_cycle+gpu_sim_cycle));
   printf("gpu_tot_issued_cta = %lld\n", gpu_tot_issued_cta);

   // performance counter for stalls due to congestion.
   printf("gpu_stall_dramfull = %d\n", gpu_stall_dramfull);
   printf("gpu_stall_icnt2sh    = %d\n", gpu_stall_icnt2sh );

   time_t curr_time;
   time(&curr_time);
   unsigned long long elapsed_time = MAX( curr_time - g_simulation_starttime, 1 );
   printf( "gpu_total_sim_rate=%u\n", (unsigned)( ( gpu_tot_sim_insn + gpu_sim_insn ) / elapsed_time ) );

   // Zhen: bottleneck
   //g_bottleneck_stats->print_average_only(stdout);
   g_bottleneck_stats->clear();

   //shader_print_l1_miss_stat( stdout );
   shader_print_cache_stats(stdout);

   cache_stats core_cache_stats;
   core_cache_stats.clear();
   for(unsigned i=0; i<m_config.num_cluster(); i++){
       m_cluster[i]->get_cache_stats(core_cache_stats);
   }
   printf("\nTotal_core_cache_stats:\n");
   core_cache_stats.print_stats(stdout, "Total_core_cache_stats_breakdown");
   shader_print_scheduler_stat( stdout, false );

   m_shader_stats->print(stdout);

/*#ifdef GPGPUSIM_POWER_MODEL
   if(m_config.g_power_simulation_enabled){
	   m_gpgpusim_wrapper->print_power_kernel_stats(gpu_sim_cycle, gpu_tot_sim_cycle, gpu_tot_sim_insn + gpu_sim_insn, kernel_info_str, true );
	   mcpat_reset_perf_count(m_gpgpusim_wrapper);
   }
#endif
*/

   // performance counter that are not local to one shader
   m_memory_stats->memlatstat_print(m_memory_config->m_n_mem,m_memory_config->nbk);
   for (unsigned i=0;i<m_memory_config->m_n_mem;i++)
      m_memory_partition_unit[i]->print(stdout);

   // L2 cache stats
   if(!m_memory_config->m_L2_config.disabled()){
       cache_stats l2_stats;
       struct cache_sub_stats l2_css;
       struct cache_sub_stats total_l2_css;
       l2_stats.clear();
       l2_css.clear();
       total_l2_css.clear();

       printf("\n========= L2 cache stats =========\n");
       for (unsigned i=0;i<m_memory_config->m_n_mem_sub_partition;i++){
           m_memory_sub_partition[i]->accumulate_L2cache_stats(l2_stats);
           m_memory_sub_partition[i]->get_L2cache_sub_stats(l2_css);

           fprintf( stdout, "L2_cache_bank[%d]: Access = %u, Miss = %u, Miss_rate = %.3lf, Pending_hits = %u, Reservation_fails = %u\n",
                    i, l2_css.accesses, l2_css.misses, (double)l2_css.misses / (double)l2_css.accesses, l2_css.pending_hits, l2_css.res_fails);

           total_l2_css += l2_css;
       }
       if (!m_memory_config->m_L2_config.disabled() && m_memory_config->m_L2_config.get_num_lines()) {
          //L2c_print_cache_stat();
          printf("L2_total_cache_accesses = %u\n", total_l2_css.accesses);
          printf("L2_total_cache_misses = %u\n", total_l2_css.misses);
          if(total_l2_css.accesses > 0)
              printf("L2_total_cache_miss_rate = %.4lf\n", (double)total_l2_css.misses/(double)total_l2_css.accesses);
          printf("L2_total_cache_pending_hits = %u\n", total_l2_css.pending_hits);
          printf("L2_total_cache_reservation_fails = %u\n", total_l2_css.res_fails);
          printf("L2_total_cache_breakdown:\n");
          l2_stats.print_stats(stdout, "L2_cache_stats_breakdown");
          total_l2_css.print_port_stats(stdout, "L2_cache");
       }
   }

   if (m_config.gpgpu_cflog_interval != 0) {
      spill_log_to_file (stdout, 1, gpu_sim_cycle);
      insn_warp_occ_print(stdout);
   }
   if ( gpgpu_ptx_instruction_classification ) {
      StatDisp( g_inst_classification_stat[g_ptx_kernel_count]);
      StatDisp( g_inst_op_classification_stat[g_ptx_kernel_count]);
   }

#ifdef GPGPUSIM_POWER_MODEL
   if(m_config.g_power_simulation_enabled){
       m_gpgpusim_wrapper->detect_print_steady_state(1,gpu_tot_sim_insn+gpu_sim_insn);
   }
#endif


   // Interconnect power stat print
   long total_simt_to_mem=0;
   long total_mem_to_simt=0;
   long temp_stm=0;
   long temp_mts = 0;
   for(unsigned i=0; i<m_config.num_cluster(); i++){
	   m_cluster[i]->get_icnt_stats(temp_stm, temp_mts);
	   total_simt_to_mem += temp_stm;
	   total_mem_to_simt += temp_mts;
   }
   printf("\nicnt_total_pkts_mem_to_simt=%ld\n", total_mem_to_simt);
   printf("icnt_total_pkts_simt_to_mem=%ld\n", total_simt_to_mem);

   time_vector_print();
   fflush(stdout);

   clear_executed_kernel_info();

   exit(0);
}


// performance counter that are not local to one shader
unsigned gpgpu_sim::threads_per_core() const 
{ 
   return m_shader_config->n_thread_per_shader; 
}

void shader_core_ctx::mem_instruction_stats(const warp_inst_t &inst)
{
    unsigned active_count = inst.active_count(); 
    //this breaks some encapsulation: the is_[space] functions, if you change those, change this.
    switch (inst.space.get_type()) {
    case undefined_space:
    case reg_space:
        break;
    case shared_space:
        m_stats->gpgpu_n_shmem_insn += active_count; 
        break;
    case const_space:
        m_stats->gpgpu_n_const_insn += active_count;
        break;
    case param_space_kernel:
    case param_space_local:
        m_stats->gpgpu_n_param_insn += active_count;
        break;
    case tex_space:
        m_stats->gpgpu_n_tex_insn += active_count;
        break;
    case global_space:
    case local_space:
        if( inst.is_store() )
            m_stats->gpgpu_n_store_insn += active_count;
        else 
            m_stats->gpgpu_n_load_insn += active_count;
        break;
    default:
        abort();
    }
}


////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Launches a cooperative thread array (CTA). 
 *  
 * @param kernel 
 *    object that tells us which kernel to ask for a CTA from 
 */

void shader_core_ctx::issue_block2core( kernel_info_t &kernel ) 
{
  //printf("I am here, in issue_block2core()\n");
    set_max_cta(kernel);

    // find a free CTA context 
    unsigned free_cta_hw_id=(unsigned)-1;
    for (unsigned i=0;i<kernel_max_cta_per_shader;i++ ) {
      if( m_cta_status[i]==0 ) {
         free_cta_hw_id=i;
         break;
      }
    }
    assert( free_cta_hw_id!=(unsigned)-1 );

    // determine hardware threads and warps that will be used for this CTA
    int cta_size = kernel.threads_per_cta();

    // hw warp id = hw thread id mod warp size, so we need to find a range 
    // of hardware thread ids corresponding to an integral number of hardware
    // thread ids
    int padded_cta_size = cta_size; 
    if (cta_size%m_config->warp_size)
      padded_cta_size = ((cta_size/m_config->warp_size)+1)*(m_config->warp_size);
    unsigned start_thread = free_cta_hw_id * padded_cta_size;
    unsigned end_thread  = start_thread +  cta_size;

    // reset the microarchitecture state of the selected hardware thread and warp contexts
    reinit(start_thread, end_thread,false);
     
    // initalize scalar threads and determine which hardware warps they are allocated to
    // bind functional simulation state of threads to hardware resources (simulation) 
    warp_set_t warps;
    unsigned nthreads_in_block= 0;
    for (unsigned i = start_thread; i<end_thread; i++) {
        m_threadState[i].m_cta_id = free_cta_hw_id;
        unsigned warp_id = i/m_config->warp_size;
        nthreads_in_block += ptx_sim_init_thread(kernel,&m_thread[i],m_sid,i,cta_size-(i-start_thread),m_config->n_thread_per_shader,this,free_cta_hw_id,warp_id,m_cluster->get_gpu());
        m_threadState[i].m_active = true; 
        warps.set( warp_id );
    }
    assert( nthreads_in_block > 0 && nthreads_in_block <= m_config->n_thread_per_shader); // should be at least one, but less than max
    m_cta_status[free_cta_hw_id]=nthreads_in_block;

    // now that we know which warps are used in this CTA, we can allocate
    // resources for use in CTA-wide barrier operations
    m_barriers.allocate_barrier(free_cta_hw_id,warps);

    // initialize the SIMT stacks and fetch hardware
    unsigned kernel_id = kernel.m_kernel_id;
    init_warps( kernel_id, free_cta_hw_id, start_thread, end_thread);
    m_n_active_cta++;
    m_n_active_tbs[kernel_id]++;

    shader_CTA_count_log(m_sid, 1);
    printf("GPGPU-Sim uArch: core:%3d, cta:%2u initialized @(%lld,%lld)\n", m_sid, free_cta_hw_id, gpu_sim_cycle, gpu_tot_sim_cycle );
    if (m_sid == 0 || m_sid == 6 || m_sid==9 || m_sid==14){
        printf("shader=%d, ", m_sid);
        for (unsigned kid = 0; kid < g_num_con_kernels; kid++) {
            printf("kernel %d TBs: %u, ", kid, m_n_active_tbs[kid]);
        }
        printf("\n");
    }
}

///////////////////////////////////////////////////////////////////////////////////////////

void dram_t::dram_log( int task ) 
{
   if (task == SAMPLELOG) {
      StatAddSample(mrqq_Dist, que_length());   
   } else if (task == DUMPLOG) {
      printf ("Queue Length DRAM[%d] ",id);StatDisp(mrqq_Dist);
   }
}

//Find next clock domain and increment its time
int gpgpu_sim::next_clock_domain(void) 
{
   double smallest = min3(core_time,icnt_time,dram_time);
   int mask = 0x00;
   if ( l2_time <= smallest ) {
      smallest = l2_time;
      mask |= L2 ;
      l2_time += m_config.l2_period;
   }
   if ( icnt_time <= smallest ) {
      mask |= ICNT;
      icnt_time += m_config.icnt_period;
   }
   if ( dram_time <= smallest ) {
      mask |= DRAM;
      dram_time += m_config.dram_period;
   }
   if ( core_time <= smallest ) {
      mask |= CORE;
      core_time += m_config.core_period;
   }
   return mask;
}

void gpgpu_sim::issue_block2core()
{
    unsigned last_issued = m_last_cluster_issue; 
    for (unsigned i=0;i<m_shader_config->n_simt_clusters;i++) {
        unsigned idx = (i + last_issued + 1) % m_shader_config->n_simt_clusters;
        unsigned num = m_cluster[idx]->issue_block2core();
        if( num ) {
            m_last_cluster_issue=idx;
            m_total_cta_launched += num;
        }
    }
}

unsigned long long g_single_step=0; // set this in gdb to single step the pipeline

void gpgpu_sim::cycle()
{
   //Added on 2017-02-23, Hongwen
   //collect Cinst/Minst/L1D-access distribution along time
   if(g_enable_runtime_stats&&g_bottleneck_stats->sampled_epoch_cycle>=1000){
        unsigned long long* new_Winst_cnt = (unsigned long long*)malloc(sizeof(unsigned long long)*MAX_CON_KERNELS);
        unsigned long long* new_Cinst_cnt = (unsigned long long*)malloc(sizeof(unsigned long long)*MAX_CON_KERNELS);
        unsigned long long* new_Minst_cnt = (unsigned long long*)malloc(sizeof(unsigned long long)*MAX_CON_KERNELS);
        unsigned long long* new_l1d_access_cnt = (unsigned long long*)malloc(sizeof(unsigned long long)*MAX_CON_KERNELS);
        
        for (unsigned kid = 0; kid < g_num_con_kernels; kid++) {
            new_Winst_cnt[kid] = g_bottleneck_stats->sampled_Winst_cnt[kid];
            new_Cinst_cnt[kid] = g_bottleneck_stats->sampled_Cinst_cnt[kid];
            new_Minst_cnt[kid] = g_bottleneck_stats->sampled_Minst_cnt[kid];
            new_l1d_access_cnt[kid] = g_bottleneck_stats->sampled_l1d_access_cnt[kid];
            
            g_bottleneck_stats->sampled_Winst_cnt[kid] = 0;
            g_bottleneck_stats->sampled_Cinst_cnt[kid] = 0;
            g_bottleneck_stats->sampled_Minst_cnt[kid] = 0;
            g_bottleneck_stats->sampled_l1d_access_cnt[kid] = 0;

   //         printf("g_bottleneck_stats->sampled_epoch_cycle=%d, new_Winst_cnt[%d]=%d, new_Cinst_cnt[%d]=%d, new_Minst_cnt[%d]=%d, new_l1d_access_cnt[%d]=%d\n", g_bottleneck_stats->sampled_epoch_cycle, kid, new_Winst_cnt[kid], kid, new_Cinst_cnt[kid], kid, new_Minst_cnt[kid], kid, new_l1d_access_cnt[kid]);
        }
      
        g_bottleneck_stats->runtime_Winst_stats.push_back(new_Winst_cnt);
        g_bottleneck_stats->runtime_Cinst_stats.push_back(new_Cinst_cnt);
        g_bottleneck_stats->runtime_Minst_stats.push_back(new_Minst_cnt);
        g_bottleneck_stats->runtime_l1d_access_stats.push_back(new_l1d_access_cnt);
        
        g_bottleneck_stats->sampled_epoch_cycle = 0;
   }

   int clock_mask = next_clock_domain();

   if (clock_mask & CORE ) {
       // shader core loading (pop from ICNT into core) follows CORE clock
      for (unsigned i=0;i<m_shader_config->n_simt_clusters;i++) 
         m_cluster[i]->icnt_cycle();
   }
    if (clock_mask & ICNT) {
        g_bottleneck_stats->icnt_cycles++;
        // pop from memory controller to interconnect
        for (unsigned i=0;i<m_memory_config->m_n_mem_sub_partition;i++) {
            mem_fetch* mf = m_memory_sub_partition[i]->top();
            if (mf) {
                unsigned response_size = mf->get_is_write()?mf->get_ctrl_size():mf->size();
                if ( ::icnt_has_buffer( m_shader_config->mem2device(i), response_size ) ) {
                    if (!mf->get_is_write()) 
                       mf->set_return_timestamp(gpu_sim_cycle+gpu_tot_sim_cycle);
                    mf->set_status(IN_ICNT_TO_SHADER,gpu_sim_cycle+gpu_tot_sim_cycle);
                    ::icnt_push( m_shader_config->mem2device(i), mf->get_tpc(), mf, response_size );
                    m_memory_sub_partition[i]->pop();
                } else {
                    gpu_stall_icnt2sh++;
                }
            } else {
               m_memory_sub_partition[i]->pop();
            }
        }
    }

   if (clock_mask & DRAM) {
      for (unsigned i=0;i<m_memory_config->m_n_mem;i++){
         m_memory_partition_unit[i]->dram_cycle(); // Issue the dram command (scheduler + delay model)
         // Update performance counters for DRAM
         m_memory_partition_unit[i]->set_dram_power_stats(m_power_stats->pwr_mem_stat->n_cmd[CURRENT_STAT_IDX][i], m_power_stats->pwr_mem_stat->n_activity[CURRENT_STAT_IDX][i],
                        m_power_stats->pwr_mem_stat->n_nop[CURRENT_STAT_IDX][i], m_power_stats->pwr_mem_stat->n_act[CURRENT_STAT_IDX][i], m_power_stats->pwr_mem_stat->n_pre[CURRENT_STAT_IDX][i],
                        m_power_stats->pwr_mem_stat->n_rd[CURRENT_STAT_IDX][i], m_power_stats->pwr_mem_stat->n_wr[CURRENT_STAT_IDX][i], m_power_stats->pwr_mem_stat->n_req[CURRENT_STAT_IDX][i]);
      }
   }

   // L2 operations follow L2 clock domain
   if (clock_mask & L2) {
      g_bottleneck_stats->l2_cycles++;
      m_power_stats->pwr_mem_stat->l2_cache_stats[CURRENT_STAT_IDX].clear();
      for (unsigned i=0;i<m_memory_config->m_n_mem_sub_partition;i++) {
          //move memory request from interconnect into memory partition (if not backed up)
          //Note:This needs to be called in DRAM clock domain if there is no L2 cache in the system
          if ( m_memory_sub_partition[i]->full() ) {
             gpu_stall_dramfull++;
          } else {
              mem_fetch* mf = (mem_fetch*) icnt_pop( m_shader_config->mem2device(i) );
              m_memory_sub_partition[i]->push( mf, gpu_sim_cycle + gpu_tot_sim_cycle );
          }
          m_memory_sub_partition[i]->cache_cycle(gpu_sim_cycle+gpu_tot_sim_cycle);
          m_memory_sub_partition[i]->accumulate_L2cache_stats(m_power_stats->pwr_mem_stat->l2_cache_stats[CURRENT_STAT_IDX]);
	  m_memory_sub_partition[i]->update_bottleneck_stats();
       }
   }

   if (clock_mask & ICNT) {
      icnt_transfer();
   }

   if (clock_mask & CORE) {
      g_bottleneck_stats->core_cycles++;
      // L1 cache + shader core pipeline stages
      m_power_stats->pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX].clear();
      for (unsigned i=0;i<m_shader_config->n_simt_clusters;i++) {
         if (m_cluster[i]->get_not_completed() || get_more_cta_left() ) {
               m_cluster[i]->core_cycle();
               *active_sms+=m_cluster[i]->get_n_active_sms();
         }
         // Update core icnt/cache stats for GPUWattch
         m_cluster[i]->get_icnt_stats(m_power_stats->pwr_mem_stat->n_simt_to_mem[CURRENT_STAT_IDX][i], m_power_stats->pwr_mem_stat->n_mem_to_simt[CURRENT_STAT_IDX][i]);
         m_cluster[i]->get_cache_stats(m_power_stats->pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX]);
      }
      float temp=0;
      for (unsigned i=0;i<m_shader_config->num_shader();i++){
        temp+=m_shader_stats->m_pipeline_duty_cycle[i];
      }
      temp=temp/m_shader_config->num_shader();
      *average_pipeline_duty_cycle=((*average_pipeline_duty_cycle)+temp);
        //cout<<"Average pipeline duty cycle: "<<*average_pipeline_duty_cycle<<endl;


      if( g_single_step && ((gpu_sim_cycle+gpu_tot_sim_cycle) >= g_single_step) ) {
          asm("int $03");
      }
      gpu_sim_cycle++;
   

      //Added on 2017-02-24, Hongwen
      g_bottleneck_stats->sampled_epoch_cycle++;
      
      if( g_interactive_debugger_enabled ) 
         gpgpu_debug();

      // McPAT main cycle (interface with McPAT)
#ifdef GPGPUSIM_POWER_MODEL
      if(m_config.g_power_simulation_enabled){
          mcpat_cycle(m_config, getShaderCoreConfig(), m_gpgpusim_wrapper, m_power_stats, m_config.gpu_stat_sample_freq, gpu_tot_sim_cycle, gpu_sim_cycle, gpu_tot_sim_insn, gpu_sim_insn);
      }
#endif

      issue_block2core();
      
      // Depending on configuration, flush the caches once all of threads are completed.
      int all_threads_complete = 1;
      if (m_config.gpgpu_flush_l1_cache) {
         for (unsigned i=0;i<m_shader_config->n_simt_clusters;i++) {
            if (m_cluster[i]->get_not_completed() == 0)
                m_cluster[i]->cache_flush();
            else
               all_threads_complete = 0 ;
         }
      }

      if(m_config.gpgpu_flush_l2_cache){
          if(!m_config.gpgpu_flush_l1_cache){
              for (unsigned i=0;i<m_shader_config->n_simt_clusters;i++) {
                  if (m_cluster[i]->get_not_completed() != 0){
                      all_threads_complete = 0 ;
                      break;
                  }
              }
          }

         if (all_threads_complete && !m_memory_config->m_L2_config.disabled() ) {
            printf("Flushed L2 caches...\n");
            if (m_memory_config->m_L2_config.get_num_lines()) {
               int dlc = 0;
               for (unsigned i=0;i<m_memory_config->m_n_mem;i++) {
                  dlc = m_memory_sub_partition[i]->flushL2();
                  assert (dlc == 0); // need to model actual writes to DRAM here
                  printf("Dirty lines flushed from L2 %d is %d\n", i, dlc  );
               }
            }
         }
      }

      if (!(gpu_sim_cycle % m_config.gpu_stat_sample_freq)) {
         time_t days, hrs, minutes, sec;
         time_t curr_time;
         time(&curr_time);
         unsigned long long  elapsed_time = MAX(curr_time - g_simulation_starttime, 1);
         if ( (elapsed_time - last_liveness_message_time) >= m_config.liveness_message_freq ) {
            days    = elapsed_time/(3600*24);
            hrs     = elapsed_time/3600 - 24*days;
            minutes = elapsed_time/60 - 60*(hrs + 24*days);
            sec = elapsed_time - 60*(minutes + 60*(hrs + 24*days));
            printf("GPGPU-Sim uArch: cycles simulated: %lld  inst.: %lld (ipc=%4.1f) sim_rate=%u (inst/sec) elapsed = %u:%u:%02u:%02u / %s", 
                   gpu_tot_sim_cycle + gpu_sim_cycle, gpu_tot_sim_insn + gpu_sim_insn, 
                   (double)gpu_sim_insn/(double)gpu_sim_cycle,
                   (unsigned)((gpu_tot_sim_insn+gpu_sim_insn) / elapsed_time),
                   (unsigned)days,(unsigned)hrs,(unsigned)minutes,(unsigned)sec,
                   ctime(&curr_time));
            fflush(stdout);
            last_liveness_message_time = elapsed_time; 
         }
         visualizer_printstat();
         m_memory_stats->memlatstat_lat_pw();
         if (m_config.gpgpu_runtime_stat && (m_config.gpu_runtime_stat_flag != 0) ) {
            if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_BW_STAT) {
               for (unsigned i=0;i<m_memory_config->m_n_mem;i++) 
                  m_memory_partition_unit[i]->print_stat(stdout);
               printf("maxmrqlatency = %d \n", m_memory_stats->max_mrq_latency);
               printf("maxmflatency = %d \n", m_memory_stats->max_mf_latency);
            }
            if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_SHD_INFO) 
               shader_print_runtime_stat( stdout );
            if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_L1MISS) 
               shader_print_l1_miss_stat( stdout );
            if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_SCHED) 
               shader_print_scheduler_stat( stdout, false );
         }
      }

      if (!(gpu_sim_cycle % 20000)) {
         // deadlock detection 
         if (m_config.gpu_deadlock_detect && gpu_sim_insn == last_gpu_sim_insn) {
            gpu_deadlock = true;
         } else {
            last_gpu_sim_insn = gpu_sim_insn;
         }
      }
      try_snap_shot(gpu_sim_cycle);
      spill_log_to_file (stdout, 0, gpu_sim_cycle);
   }


   // skip first 10k cycles to warm up
   //if (gpu_sim_cycle == 20000)
   //  g_bottleneck_stats->clear();

   if (m_config.bottleneck_stats_period != 0) {
     unsigned period = m_config.bottleneck_stats_period;
     if ((gpu_sim_cycle % period) == (period-1)
	 && g_bottleneck_stats->icnt_cycles > period/2) {
       std::string fname = "bottleneck_stats" + m_executed_kernel_names[0] + ".out";
       FILE *ofile = fopen(fname.c_str(), "a");
       //g_bottleneck_stats->print(ofile);
       //g_bottleneck_stats->print_average_only(ofile);
       g_bottleneck_stats->print_con_kernels(ofile);
       g_bottleneck_stats->clear();
       fclose(ofile);
     }
   }
   if (m_config.gpgpu_limit_cycles != 0) {
     if (gpu_sim_cycle+gpu_tot_sim_cycle >= m_config.gpgpu_limit_cycles) {
       print_stats();
       printf("\nReached limit cycles %llu\n", m_config.gpgpu_limit_cycles);
       exit(0);
     }
   }
   if (m_config.gpgpu_limit_insns != 0) {
     if (gpu_tot_sim_insn+gpu_sim_insn >= m_config.gpgpu_limit_insns) {
       print_stats();
       printf("\nReached limit instructions %llu\n", m_config.gpgpu_limit_insns);
       exit(0);
     }
   }
}


void shader_core_ctx::dump_warp_state( FILE *fout ) const
{
   fprintf(fout, "\n");
   fprintf(fout, "per warp functional simulation status:\n");
   for (unsigned w=0; w < m_config->max_warps_per_shader; w++ ) 
       m_warp[w].print(fout);
}

void gpgpu_sim::dump_pipeline( int mask, int s, int m ) const
{
/*
   You may want to use this function while running GPGPU-Sim in gdb.
   One way to do that is add the following to your .gdbinit file:
 
      define dp
         call g_the_gpu.dump_pipeline_impl((0x40|0x4|0x1),$arg0,0)
      end
 
   Then, typing "dp 3" will show the contents of the pipeline for shader core 3.
*/

   printf("Dumping pipeline state...\n");
   if(!mask) mask = 0xFFFFFFFF;
   for (unsigned i=0;i<m_shader_config->n_simt_clusters;i++) {
      if(s != -1) {
         i = s;
      }
      if(mask&1) m_cluster[m_shader_config->sid_to_cluster(i)]->display_pipeline(i,stdout,1,mask & 0x2E);
      if(s != -1) {
         break;
      }
   }
   if(mask&0x10000) {
      for (unsigned i=0;i<m_memory_config->m_n_mem;i++) {
         if(m != -1) {
            i=m;
         }
         printf("DRAM / memory controller %u:\n", i);
         if(mask&0x100000) m_memory_partition_unit[i]->print_stat(stdout);
         if(mask&0x1000000)   m_memory_partition_unit[i]->visualize();
         if(mask&0x10000000)   m_memory_partition_unit[i]->print(stdout);
         if(m != -1) {
            break;
         }
      }
   }
   fflush(stdout);
}

const struct shader_core_config * gpgpu_sim::getShaderCoreConfig()
{
   return m_shader_config;
}

const struct memory_config * gpgpu_sim::getMemoryConfig()
{
   return m_memory_config;
}

simt_core_cluster * gpgpu_sim::getSIMTCluster()
{
   return *m_cluster;
}

void bottleneck_stats::clear() {

  dram_n_cmd = 0;
  icnt_cycles = 0;
  core_cycles = 0;
  l2_cycles = 0;
  n_gmem_load_insns = 0;
  n_gmem_load_accesses = 0;
  n_smem_access_insn = 0;
  n_smem_accesses = 0;
  n_gmem_load_useful_bytes = 0;
  n_gmem_load_transaction_bytes = 0;
  
  schd_fetch = 0;
  schd_sync = 0;
  schd_control_hzd = 0;
  schd_data_hzd = 0;
  schd_struct_hzd = 0;
  schd_run = 0;
  //tmp_counter = 0;
  dram_activity = 0;
  
  n_1ld_mshrs_full = 0;
  n_l1d_missq_full = 0;
  
  //Added on 2017-02-24, Hongwen
  //record runtime statistics
  sampled_epoch_cycle = 0;
  for(unsigned i=0;i<runtime_Winst_stats.size();i++){
    delete[] runtime_Winst_stats[i];
    delete[] runtime_Cinst_stats[i];
    delete[] runtime_Minst_stats[i];
    delete[] runtime_l1d_access_stats[i];
  }
    runtime_Winst_stats.clear();
    runtime_Cinst_stats.clear();
    runtime_Minst_stats.clear();
    runtime_l1d_access_stats.clear();

  for(unsigned set_index=0;set_index<32;set_index++)
        L1D_access_set[set_index] = 0;
  
  MDB_global_keep_num = -1;
  for (unsigned i = 0; i < MAX_CON_KERNELS; i++) {
    n_l1d_bypass_num[i] = 0;
    bypass_l1d_request_latency[i] = 0;
    access_l1d_request_latency[i] = 0;
    n_l1d_bp_rsfails[i] = 0;
    n_l1d_max_bp_in_circle_num[i] = 0;
    n_l1d_accu_bp_in_circle_num[i] = 0;
    n_l1d_accu_bp_in_circle_cycles[i] = 0;
    //Added on 2017-02-11, Hongwen
    Cinst_cnt_kernel[i] = 0;
    Minst_cnt_kernel[i] = 0;
    Minst_Smem_cnt_kernel[i] = 0;
    Minst_Ccache_cnt_kernel[i] = 0;
    Minst_Tcache_cnt_kernel[i] = 0;
    Minst_Dcache_cnt_kernel[i] = 0;
    //access, bypass, L1D_hit, L1D_miss
    for(unsigned shader_id=0; shader_id<16; shader_id++){
        MDB_sampled_rsfails[shader_id] = 0;//reset sampled rsfails
        Minst_Dcache_cnt_kernel_shader[shader_id][i] = 0;
        n_l1d_access_kernel_shader[shader_id][i] = 0;
        //for memory instruction limiting
        Inflight_Cinst_accu_kernel_shader[shader_id][i] = 0;
        Inflight_Minst_accu_kernel_shader[shader_id][i] = 0;
  
        sampled_Winst_cnt[i] = 0;
        sampled_Cinst_cnt[i] = 0;
        sampled_Minst_cnt[i] = 0;
        sampled_l1d_access_cnt[i] = 0;
  }
    n_l1d_slot_resfail[i] = 0;
    n_l1d_mshrs_resfail[i] = 0;
    n_l1d_missq_resfail[i] = 0;
    
  
    //sampled_Winst_cnt_kernel[i] = 0;
    Winst_quota_kernel[i] = 0;
  
  //Added on 2017-02-01, Hongwen
    inst_stats_kernel[i].clear();
    cache_block_access_stats_kernel[i].clear();
    access_cnt_based_stats_kernel[i].clear();
    
    dram_bwutil[i] = 0;
    l2_mshr_util[i] = 0;
    icnt_s2m_injected_flits[i] = 0;
    icnt_m2s_injected_flits[i] = 0;
    l1_mshr_util[i] = 0;
    scheduler_util[i] = 0;
    mem_pipe_util[i] = 0;
    sp_pipe_util[i] = 0;
    sfu_pipe_util[i] = 0;
    smem_cycles[i] = 0;
    //reg_util[i] = 0;
    thread_insn[i] = 0;
    l1_missq_util[i] = 0;
    l2_missq_util[i] = 0;
    //l1_pipe_util[i] = 0;
    //ldst_memory_cycle[i] = 0;
    l1_tag_util[i] = 0;
    l1_data_util[i] = 0;
    l2_tag_util[i] = 0;
    l2_data_util[i] = 0;
    l2_fill_util[i] = 0;
    l1_fill_util[i] = 0;
    //l1i_tag_util[i] = 0;
    //ldst_writeback[i] = 0;
    n_l1_access[i] = 0;
    n_l1_hit[i] = 0;
    n_l1_miss[i] = 0;
    n_l1_resfail[i] = 0;
    n_l2_access[i] = 0;
    n_l2_hit[i] = 0;
    n_l2_miss[i] = 0;
    n_l2_resfail[i] = 0;
    sm_reg_util[i] = 0;
    sm_smem_util[i] = 0;
    sm_tb_util[i] = 0;
    sm_thread_util[i] = 0;
    icnt_s2m_total_packets[i] = 0;
    icnt_m2s_total_packets[i] = 0;

    //q_icnt_l2_empty[i] = 0;
    //q_icnt_l2_full[i] = 0;
    //q_icnt_mem_vc_full[i] = 0;
    //q_l2_dram_full[i] = 0;
    //q_rop_empty[i] = 0;
    //q_l2_dram_unempty[i] = 0;

  }
}
/*
void bottleneck_stats::compute_stats(const unsigned long long *stats, unsigned *maxid, unsigned long long *max, unsigned *nzeros, unsigned long long *sum) const {
  *max = 0;
  *sum = 0;
  *nzeros = 0;
  for (unsigned i = 0; i < MAX_UNITS; i++) {
    *sum += stats[i];
    if (stats[i] != 0)
      (*nzeros)++;
    if (stats[i] > *max) {
      *maxid = i;
      *max = stats[i];
    }
  }
}
*/
void bottleneck_stats::print_item_vs(FILE *outf, const char* name, const unsigned long long *util, unsigned long long denom) const {
  assert(denom != 0);
  fprintf(outf, "%s\t", name);
  double total = .0;
  for (unsigned k = 0; k < g_num_con_kernels; k++) {
    double u = (double)util[k] / (double)denom;
    total += u;
    fprintf(outf, "%.3lf\t", u);
  }
  fprintf(outf, "%.3lf\n", total);
}
void bottleneck_stats::print_item_vv(FILE *outf, const char* name, const unsigned long long *util, const unsigned long long *denom) const {
  assert(denom != 0);
  fprintf(outf, "%s\t", name);
  double total = .0;
  long long tot_util = 0;
  long long tot_denom = 0;
  for (unsigned k = 0; k < g_num_con_kernels; k++) {
    double u = (double)util[k] / (double)denom[k];
    //total += u;
    tot_util += util[k];
    tot_denom += denom[k];
    fprintf(outf, "%.3lf\t", u);
  }
  total = (double)tot_util/(double)tot_denom;
  fprintf(outf, "%.3lf\n", total);
}

void bottleneck_stats::print_item_ss(FILE *outf, const char* name, unsigned long long util, unsigned long long denom) const {
  assert(denom != 0);
  double u = (double)util / (double)denom;
  fprintf(outf, "%s\t%.3lf\n", name, u);
}

void bottleneck_stats::normalize_vector(const double *in, double *out, unsigned critical) const 
{
  assert(critical < 3);
  assert(in[critical] > 1.0e-3);
  out[0] = in[0] / in[critical];
  out[1] = in[1] / in[critical];
  out[2] = in[2] / in[critical];
}


/*
double bottleneck_stats::get_vector_distance(const double *v1, const double *v2, unsigned size) const {
  double result = 0.0;
  for (unsigned i = 0; i < size; i++) {
    result += pow(v1[i] - v2[i], 2.0);
  }
  return result;
}
*/

unsigned bottleneck_stats::max_index3(const double *in3) const {
  unsigned imax;
  imax = in3[0] > in3[1] ? 0 : 1;
  imax = in3[imax] > in3[2] ? imax : 2;
  return imax;
}

unsigned bottleneck_stats::min_index3(const double *in3) const {
  unsigned imin;
  imin = in3[0] < in3[1] ? 0 : 1;
  imin = in3[imin] < in3[2] ? imin : 2;
  return imin;
}

unsigned bottleneck_stats::find_low_resource(const double *array3, const bool *pcr) const {
  unsigned low = (unsigned)-1;
  double low_res = 9999999.9;
  for (unsigned i = 0; i < 3; i++) {
    if (pcr[i] && array3[i] < low_res) {
      low = i;
      low_res = array3[i];
    }
  }
  assert(low != (unsigned)-1);
  return low;
}

double bottleneck_stats::get_advance_factor(const double *nutil, const double *nremain) const {
  unsigned imax = max_index3(nutil);
  return nutil[imax] * nremain[imax];
}

void bottleneck_stats::get_kernel_factors(double *advance, double *retreat) const {
  unsigned n_sm = g_the_gpu->m_config.num_shader();
  unsigned n_schedulers = g_the_gpu->m_shader_config->gpgpu_num_sched_per_core;
  unsigned n_sub_mem = g_the_gpu->m_memory_config->m_n_mem_sub_partition;
  unsigned n_mem = g_the_gpu->m_memory_config->m_n_mem;

  // scheduler, icnt, dram
  double kernel_vec[MAX_CON_KERNELS][3] = {0.0};
  double normal_vec[MAX_CON_KERNELS][3] = {0.0};
  double remain_vec[3] = {1.0, 1.0, 1.0};
  double normal_remain_vec[3];
  double sum_vec[3] = {0.0};
  double intensive_vec[3] = {0.0};
  // potential critical resources
  bool pcr[3] = {false};

  for (unsigned k = 0; k < g_num_con_kernels; k++) {
    double schd = (double)scheduler_util[k] / (n_sm*core_cycles*n_schedulers*1.0);
    double icnt = (double)icnt_m2s_injected_flits[k] / (icnt_cycles*n_sub_mem*.6);
    double dram = (double)dram_bwutil[k] / (n_mem*dram_n_cmd*.75);
    sum_vec[0] += schd;
    sum_vec[1] += icnt;
    sum_vec[2] += dram;
    kernel_vec[k][0] = schd;
    kernel_vec[k][1] = icnt;
    kernel_vec[k][2] = dram;
    unsigned critical = max_index3(kernel_vec[k]);
    pcr[critical] = true;
  }

  /*  
  remain_vec[0] -= sum_vec[0];
  remain_vec[1] -= sum_vec[1];
  remain_vec[2] -= sum_vec[2];
  assert(remain_vec[0] > 0 && remain_vec[1] > 0 && remain_vec[2] > 0);
  */

  unsigned high_res = max_index3(sum_vec);
  unsigned low_res = find_low_resource(sum_vec, pcr);

  for (unsigned k = 0; k < g_num_con_kernels; k++) {
    //double adv = get_advance_factor(normal_vec[k], normal_remain_vec);
    double adv = kernel_vec[k][low_res] / kernel_vec[k][high_res];
    advance[k] = adv;
    double ret = kernel_vec[k][high_res];
    retreat[k] = ret;
  }
}

void bottleneck_stats::print_kernel_factors(FILE *outf, double *advance, double *retreat) const {
  fprintf(outf, "advance factor\t");
  for (unsigned k = 0; k < g_num_con_kernels; k++) {
    fprintf(outf, "%.3lf\t", advance[k]);
  }
  fprintf(outf, "\n");

  fprintf(outf, "retreat factor\t");
  for (unsigned k = 0; k < g_num_con_kernels; k++) {
    fprintf(outf, "%.3lf\t", retreat[k]);
  }
  fprintf(outf, "\n");
}
    
void bottleneck_stats::print_con_kernel_runtime_stats(FILE *outf, std::vector<unsigned long long*> runtime_stats) const{
    for(unsigned i=0;i<runtime_stats.size();i++){
//    for(std::vector<unsigned long long*>::iterator it = runtime_stats.begin() ; it != runtime_stats.end(); ++it){
        fprintf(outf, "%d ", i);
        print_item_vs(outf, "", runtime_stats[i], 1);
    }
}

void bottleneck_stats::print_con_kernels(FILE *outf) const {
  assert(outf);
  extern gpgpu_sim *g_the_gpu;
  unsigned n_sm = g_the_gpu->m_config.num_shader();
  unsigned n_mem = g_the_gpu->m_memory_config->m_n_mem;
  unsigned n_sub_mem = g_the_gpu->m_memory_config->m_n_mem_sub_partition;
  unsigned n_node = n_sm + n_sub_mem;
  unsigned n_l1d_mshr = g_the_gpu->m_shader_config->m_L1D_config.get_num_mshr();
  unsigned n_l1d_missq = g_the_gpu->m_shader_config->m_L1D_config.get_missq_size();
  unsigned n_l1t_mshr = g_the_gpu->m_shader_config->m_L1T_config.get_num_mshr();
  unsigned n_l2_mshr = g_the_gpu->m_memory_config->m_L2_config.get_num_mshr();
  unsigned n_l2_missq = g_the_gpu->m_memory_config->m_L2_config.get_missq_size();
  unsigned n_schedulers = g_the_gpu->m_shader_config->gpgpu_num_sched_per_core;
  unsigned n_mem_pipe = g_the_gpu->m_shader_config->gpgpu_num_mem_units;
  unsigned n_sp_pipe = g_the_gpu->m_shader_config->gpgpu_num_sp_units;
  unsigned n_sfu_pipe = g_the_gpu->m_shader_config->gpgpu_num_sfu_units;
  unsigned n_reg_bank = g_the_gpu->m_shader_config->gpgpu_num_reg_banks;
  unsigned n_thread = g_the_gpu->m_shader_config->n_thread_per_shader;
  unsigned n_tb = g_the_gpu->m_shader_config->max_cta_per_core;
  unsigned n_reg = g_the_gpu->m_shader_config->gpgpu_shader_registers;
  unsigned n_smem = g_the_gpu->m_shader_config->gpgpu_shmem_size;

  //const struct memory_config *mem_config = g_the_gpu->m_memory_config;
  fprintf(outf, "%s", g_the_gpu->executed_kernel_info_string().c_str());
  fprintf(outf, "cycles: core %llu, icnt %llu, l2 %llu, dram %llu\n", core_cycles, icnt_cycles, l2_cycles, dram_n_cmd);
  fprintf(outf, "Kernel_max_TBs: %u %u %u %u\n", g_con_kernel_max_tbs[0], g_con_kernel_max_tbs[1], g_con_kernel_max_tbs[2], g_con_kernel_max_tbs[3]);
  fprintf(outf, "############## bottleneck_stats #############\n");

  fprintf(outf, "component\t");
  for (unsigned k = 0; k < g_num_con_kernels; k++) {
    fprintf(outf, "kernel%u\t", k+1);
  }
  fprintf(outf, "total\n");
  
  print_item_vs(outf, "gpu ipc   ", thread_insn, core_cycles);

  fprintf(outf, "\n");
    
  //Added on 2017-02-11, Hongwen
  //statistics for Hongwen
  fprintf(outf, "############## Hongwen_stats #############\n");
  print_item_vs(outf, "Cinst cnt", Cinst_cnt_kernel, 1);
  print_item_vs(outf, "Minst cnt", Minst_cnt_kernel, 1);
  print_item_vs(outf, "Minst_Smem cnt", Minst_Smem_cnt_kernel, 1);
  print_item_vs(outf, "Minst_Ccache cnt", Minst_Ccache_cnt_kernel, 1);
  print_item_vs(outf, "Minst_Tcache cnt", Minst_Tcache_cnt_kernel, 1);
  print_item_vs(outf, "Minst_Dcache cnt", Minst_Dcache_cnt_kernel, 1);
  
  unsigned long long tot_Minst_Dcache_cnt_k0 = 0;
  for(unsigned shader_id=0;shader_id<16; shader_id++){
    fprintf(outf, "shader %d ", shader_id);
    print_item_vs(outf, "Minst_Dcache cnt", Minst_Dcache_cnt_kernel_shader[shader_id], 1);
    tot_Minst_Dcache_cnt_k0 += Minst_Dcache_cnt_kernel_shader[shader_id][0];
  }
  fprintf(outf, "tot Minst_Dcache_k0 cnt %d\n", tot_Minst_Dcache_cnt_k0);
  
  print_item_vs(outf, "Minst_other cnt", Minst_other_cnt_kernel, 1);
  print_item_vs(outf, "Load inst", load_inst_kernel, 1);
  print_item_vs(outf, "Store inst", store_inst_kernel, 1);
  print_item_vs(outf, "Barrier inst", barrier_inst_kernel, 1);
  print_item_vs(outf, "l1d accesses", n_l1_access, 1);
  print_item_vs(outf, "l1d bypasses", n_l1d_bypass_num, 1);
  print_item_vv(outf, "avg bypass_l1d_request latency", bypass_l1d_request_latency, n_l1d_bypass_num); 
  print_item_vv(outf, "avg access_l1d_request latency", access_l1d_request_latency, n_l1_access); 
  print_item_vs(outf, "l1d bypass_rsfails", n_l1d_bp_rsfails, 1);
  print_item_vs(outf, "max l1d bp_in_circle", n_l1d_max_bp_in_circle_num, 1);
  print_item_vv(outf, "avg l1d bp_in_circle", n_l1d_accu_bp_in_circle_num, n_l1d_accu_bp_in_circle_cycles);
  print_item_vs(outf, "l2 accesses", n_l2_access, 1);
  
  /*
  for(unsigned shader_id=0;shader_id<16; shader_id++){
    fprintf(outf, "shader %d ", shader_id);
    print_item_vs(outf, "l1d accesses", n_l1d_access_kernel_shader[shader_id], 1);
  }
  */
  
  for(unsigned set_index=0;set_index<32;set_index++)
    fprintf(outf, "l1d access_set-%d %d\n", set_index, L1D_access_set[set_index]);
  
  print_item_vv(outf, "Cinst/Minst", Cinst_cnt_kernel, Minst_cnt_kernel);
  print_item_vv(outf, "Req/Minst_cache", n_l1_access, Minst_Dcache_cnt_kernel);
  print_item_vv(outf, "Cinst/Req", Cinst_cnt_kernel, n_l1_access);
  print_item_vs(outf, "l1d hits", n_l1_hit, 1);
  print_item_vs(outf, "l1d misses", n_l1_miss, 1);
  print_item_vs(outf, "l1d rsfails", n_l1_resfail, 1);
  print_item_vv(outf, "l1d hit rate", n_l1_hit, n_l1_access);
  print_item_vv(outf, "l1d miss rate", n_l1_miss, n_l1_access);
  print_item_vv(outf, "l1d rsfail rate", n_l1_resfail, n_l1_access);
  print_item_vs(outf, "l1d slot rsfail", n_l1d_slot_resfail, 1);
  print_item_vs(outf, "l1d mshr rsfail", n_l1d_mshrs_resfail, 1);
  print_item_vs(outf, "l1d missq rsfail", n_l1d_missq_resfail, 1);
  
  /*
  for(unsigned shader_id=0;shader_id<16; shader_id++){
    fprintf(outf, "shader %d ", shader_id);
    print_item_vs(outf, "inflight avg_Cinst_cnt", Inflight_Cinst_accu_kernel_shader[shader_id], core_cycles);
  }
  for(unsigned shader_id=0;shader_id<16; shader_id++){
    fprintf(outf, "shader %d ", shader_id);
    print_item_vs(outf, "inflight avg_Minst_cnt", Inflight_Minst_accu_kernel_shader[shader_id], core_cycles);
  }
  */
  print_item_vs(outf, "global inflight avg_Cinst_cnt", Inflight_Cinst_accu_kernel, n_sm*core_cycles);
  print_item_vs(outf, "global inflight avg_Minst_cnt", Inflight_Minst_accu_kernel, n_sm*core_cycles);
  
  //fprintf(outf, "l1d mshr_full cycle %d ", ceil((float)n_1ld_mshrs_full/16));
  //fprintf(outf, "l1d missq_full cycle %d ", ceil((float)n_l1d_missq_full/16));
  fprintf(outf, "l1d mshr_full cycle %d\n", n_1ld_mshrs_full);
  fprintf(outf, "l1d missq_full cycle %d\n", n_l1d_missq_full);
  
  fprintf(outf, "l1d mshr_full fraction %.3lf\n", (float)n_1ld_mshrs_full/n_sm/core_cycles);
  fprintf(outf, "l1d missq_full fraction %.3lf\n", (float)n_l1d_missq_full/n_sm/core_cycles);
  
  fprintf(outf, "############## end_Hongwen_stats #############\n");
  
  // TLP related
  print_item_vs(outf, "reg file", sm_reg_util, (unsigned long long)n_reg);
  print_item_vs(outf, "smem file", sm_smem_util, n_smem);
  print_item_vs(outf, "thread slot", sm_thread_util, n_thread);
  print_item_vs(outf, "TB slot  ", sm_tb_util, n_tb);

  fprintf(outf, "\n");
  // mostly used
  print_item_vs(outf, "s util", scheduler_util, n_sm*core_cycles*n_schedulers);
  print_item_vs(outf, "i util", icnt_m2s_injected_flits, icnt_cycles*n_sub_mem*.6);
  print_item_vs(outf, "d util", dram_bwutil, n_mem*dram_n_cmd*.75);

  double advance_factor[MAX_CON_KERNELS] = {0.0};
  double retreat_factor[MAX_CON_KERNELS] = {0.0};
  get_kernel_factors(advance_factor, retreat_factor);
  print_kernel_factors(outf, advance_factor, retreat_factor);
  
  fprintf(outf, "\n");

  // CORE
  print_item_vs(outf, "scheduler util", scheduler_util, n_sm*core_cycles*n_schedulers);
  print_item_vs(outf, "smem util", smem_cycles, n_sm*core_cycles);
  print_item_vs(outf, "mem pipe util", mem_pipe_util, n_sm*core_cycles*n_mem_pipe);
  print_item_vs(outf, "sp pipe util", sp_pipe_util, n_sm*core_cycles*n_sp_pipe);
  print_item_vs(outf, "sfu pipe util", sfu_pipe_util, n_sm*core_cycles*n_sfu_pipe);
  
  // L1D
  print_item_vs(outf, "l1d tag util", l1_tag_util, n_sm*core_cycles);
  print_item_vs(outf, "l1d data util", l1_data_util, n_sm*core_cycles);
  print_item_vs(outf, "l1d fill util", l1_fill_util, n_sm*core_cycles);
  print_item_vs(outf, "l1d mshr util", l1_mshr_util, n_sm*core_cycles*n_l1d_mshr);
  print_item_vs(outf, "l1d missq util", l1_missq_util, n_sm*core_cycles*n_l1d_missq);
  print_item_vv(outf, "l1d hit rate", n_l1_hit, n_l1_access);
  print_item_vv(outf, "l1d miss rate", n_l1_miss, n_l1_access);
  print_item_vv(outf, "l1d rsfail rate", n_l1_resfail, n_l1_access);

  // icnt
  print_item_vs(outf, "icnt s2m util", icnt_s2m_injected_flits, icnt_cycles*n_sm*.6);
  print_item_vv(outf, "icnt s2m fpp", icnt_s2m_injected_flits, icnt_s2m_total_packets);
  print_item_vs(outf, "icnt m2s util", icnt_m2s_injected_flits, icnt_cycles*n_sub_mem*.6);
  print_item_vv(outf, "icnt m2s fpp", icnt_m2s_injected_flits, icnt_m2s_total_packets);

  // L2
  print_item_vs(outf, "l2 tag util", l2_tag_util, l2_cycles*n_sub_mem);
  print_item_vs(outf, "l2 data util", l2_data_util, l2_cycles*n_sub_mem);
  print_item_vs(outf, "l2 fill util", l2_fill_util, l2_cycles*n_sub_mem);
  print_item_vs(outf, "l2 mshr util", l2_mshr_util, l2_cycles*n_sub_mem*n_l2_mshr);
  print_item_vs(outf, "l2 missq util", l2_missq_util, l2_cycles*n_sub_mem*n_l2_missq);
  print_item_vv(outf, "l2 hit rate", n_l2_hit, n_l2_access);
  print_item_vv(outf, "l2 miss rate", n_l2_miss, n_l2_access);
  print_item_vv(outf, "l2 rsfail rate", n_l2_resfail, n_l2_access);

  // dram
  print_item_vs(outf, "dram util", dram_bwutil, n_mem*dram_n_cmd);
  print_item_ss(outf, "dram activity", dram_activity, n_mem*dram_n_cmd);

  fprintf(outf, "load trans eff\t%.3lf\n", (double)n_gmem_load_useful_bytes/n_gmem_load_transaction_bytes);
  fprintf(outf, "load trans sz\t%.3lf\n", (double)n_gmem_load_transaction_bytes/n_gmem_load_accesses);

  fprintf(outf, "\n");

  unsigned long long schd_total = schd_fetch + schd_sync + schd_control_hzd + schd_data_hzd + schd_struct_hzd + schd_run;
  fprintf(outf, "run %.3lf, fetch %.3lf, sync %.3lf, control %.3lf, data %.3lf, struct %.3lf\n", (double)schd_run/schd_total, (double)schd_fetch/schd_total, (double)schd_sync/schd_total, (double)schd_control_hzd/schd_total, (double)schd_data_hzd/schd_total, (double)schd_struct_hzd/schd_total);
  fprintf(outf, "############ end_bottleneck_stats #############\n");

}

void bottleneck_stats::print_icache_stats(FILE *outf, unsigned icache_line_size, unsigned icache_size) const {
                //if(m_sid==1 && g_bottleneck_stats->inst_stats_kernel[0].size()==41){
                    for(unsigned i=0;i<MAX_CON_KERNELS;i++){
                        fprintf(outf, "kernel-%d:\n",i);
                        unsigned tot_inst_cnt = 0;  
                        unsigned tot_inst_size = 0;
                        unsigned tot_cache_line_cnt = 0;
                        unsigned tot_cache_line_size = 0;
                        unsigned tot_access_cnt = 0;
                        unsigned tot_hit_cnt = 0;
                        unsigned tot_miss_cnt = 0;
                        unsigned tot_rf_cnt = 0;
                        float tot_miss_rate = 0.0;
                        float tot_rf_per_access = 0.0;
                        
                        //from individual instruction stats to block_id based stats
                        //pc, ptx_inst, access_cnt, hit_cnt, miss_cnt, rf_cnt
                        for (std::map<unsigned, struct accessed_inst_info>::iterator it=g_bottleneck_stats->inst_stats_kernel[i].begin(); it!=g_bottleneck_stats->inst_stats_kernel[i].end(); ++it){
                            
                            //block_id -> access_cnt
                            unsigned cache_block_id = it->second.cache_block_id;
                            if(g_bottleneck_stats->cache_block_access_stats_kernel[i].find(cache_block_id) == g_bottleneck_stats->cache_block_access_stats_kernel[i].end()){
                                //access_cnt, hit_cnt, miss_cnt, rf_cnt, miss_rate, rf_per_access
                                struct accessed_cache_block_info new_cache_block_info = {0, 0, 0, 0, 0.0, 0.0};
                                std::pair<unsigned, struct accessed_cache_block_info> new_block_access_info(cache_block_id, new_cache_block_info);
                                g_bottleneck_stats->cache_block_access_stats_kernel[i].insert(new_block_access_info);
                            }
                            g_bottleneck_stats->cache_block_access_stats_kernel[i][cache_block_id].access_cnt += it->second.access_cnt;
                            g_bottleneck_stats->cache_block_access_stats_kernel[i][cache_block_id].hit_cnt += it->second.hit_cnt;
                            g_bottleneck_stats->cache_block_access_stats_kernel[i][cache_block_id].miss_cnt += it->second.miss_cnt;
                            g_bottleneck_stats->cache_block_access_stats_kernel[i][cache_block_id].rf_cnt += it->second.rf_cnt;
                        
                        }
                        tot_inst_cnt = g_bottleneck_stats->inst_stats_kernel[i].size()*2;
                        if(tot_inst_cnt>0)
                            tot_inst_size = g_bottleneck_stats->inst_stats_kernel[i].begin()->second.ptx_inst_size_1 * tot_inst_cnt;
                        
                        //from block_id based stats access_cnt based stats
                        tot_cache_line_cnt = g_bottleneck_stats->cache_block_access_stats_kernel[i].size();
                        tot_cache_line_size = tot_cache_line_cnt*icache_line_size;
                        for (std::map<unsigned, struct accessed_cache_block_info>::iterator it=g_bottleneck_stats->cache_block_access_stats_kernel[i].begin(); it!=g_bottleneck_stats->cache_block_access_stats_kernel[i].end(); ++it){
                            //access_cnt -> stats
                            unsigned access_cnt = it->second.access_cnt;
                            if(g_bottleneck_stats->access_cnt_based_stats_kernel[i].find(access_cnt) == g_bottleneck_stats->access_cnt_based_stats_kernel[i].end()){
                                //block_cnt, hit_cnt, miss_cnt, rf_cnt, miss_rate, rf_per_access
                                struct access_cnt_based_info new_cnt_based_info = {0, 0, 0, 0, 0.0, 0.0};
                                std::pair<unsigned, struct access_cnt_based_info> new_access_cnt_based_info(access_cnt, new_cnt_based_info);
                                g_bottleneck_stats->access_cnt_based_stats_kernel[i].insert(new_access_cnt_based_info);
                            }
                            g_bottleneck_stats->access_cnt_based_stats_kernel[i][access_cnt].block_cnt++;
                            g_bottleneck_stats->access_cnt_based_stats_kernel[i][access_cnt].hit_cnt += it->second.hit_cnt;
                            g_bottleneck_stats->access_cnt_based_stats_kernel[i][access_cnt].miss_cnt += it->second.miss_cnt;
                            g_bottleneck_stats->access_cnt_based_stats_kernel[i][access_cnt].rf_cnt += it->second.rf_cnt;
                        }

                            for (std::map<unsigned, struct access_cnt_based_info>::iterator it=g_bottleneck_stats->access_cnt_based_stats_kernel[i].begin(); it!=g_bottleneck_stats->access_cnt_based_stats_kernel[i].end(); ++it){
                                tot_access_cnt += it->first * it->second.block_cnt;
                                tot_hit_cnt += it->second.hit_cnt;
                                tot_miss_cnt += it->second.miss_cnt;
                                tot_rf_cnt += it->second.rf_cnt;
                            }
                            tot_miss_rate = (float)tot_miss_cnt/tot_access_cnt;
                            tot_rf_per_access = (float)tot_rf_cnt/tot_access_cnt;
                        
                        //print overall stats
                        float ratio_working_set_over_cache_size = (float)tot_cache_line_size/icache_size;
                        float ratio_inst_size_over_cache_line_size = (float)tot_inst_size/tot_cache_line_size;
                        fprintf(outf, "overall stats:\n",i);
                        fprintf(outf, "icache_size, tot_cache_line_size, tot_inst_size, ratio_working_set_over_cache_size, ratio_inst_size_over_cache_line_size, tot_cache_line_cnt, tot_inst_cnt, tot_miss_rate, tot_rf_per_access\n");
                        fprintf(outf, "%d, %d, %d, %.3f, %.3f, %d, %d, %.3f, %.3f\n", icache_size, tot_cache_line_size, tot_inst_size, ratio_working_set_over_cache_size, ratio_inst_size_over_cache_line_size, tot_cache_line_cnt, tot_inst_cnt, tot_miss_rate, tot_rf_per_access);
                        
                        //print cache block stats
                        fprintf(outf, "cache_block_id based stats:\n",i);
                        fprintf(outf, "block_id, access_cnt, hit_cnt, miss_cnt, rf_cnt, miss_rate, rf_per_access\n",i);
                        for (std::map<unsigned, struct accessed_cache_block_info>::iterator it=g_bottleneck_stats->cache_block_access_stats_kernel[i].begin(); it!=g_bottleneck_stats->cache_block_access_stats_kernel[i].end(); ++it){
                            it->second.miss_rate = (float)it->second.miss_cnt/it->second.access_cnt;
                            it->second.rf_per_access = (float)it->second.rf_cnt/it->second.access_cnt;
                            fprintf(outf, "%d, %d, %d, %d, %d, %.3f, %.3f\n", it->first, it->second.access_cnt, it->second.hit_cnt, it->second.miss_cnt, it->second.rf_cnt, it->second.miss_rate, it->second.rf_per_access);
                        }
                            
                        fprintf(outf, "access_cnt based stats:\n",i);
                        fprintf(outf, "access_cnt, block_cnt, hit_cnt, miss_cnt, rf_cnt, miss_rate, rf_per_access\n",i);
                        for (std::map<unsigned, struct access_cnt_based_info>::iterator it=g_bottleneck_stats->access_cnt_based_stats_kernel[i].begin(); it!=g_bottleneck_stats->access_cnt_based_stats_kernel[i].end(); ++it){
                            it->second.miss_rate = (float)it->second.miss_cnt/(it->first*it->second.block_cnt);
                            it->second.rf_per_access = (float)it->second.rf_cnt/(it->first*it->second.block_cnt);
                            fprintf(outf, "%d, %d, %d, %d, %d, %.3f, %.3f\n", it->first, it->second.block_cnt, it->second.hit_cnt, it->second.miss_cnt, it->second.rf_cnt, it->second.miss_rate, it->second.rf_per_access);
                        }
                    
                        for (std::map<unsigned, struct accessed_inst_info>::iterator it=g_bottleneck_stats->inst_stats_kernel[i].begin(); it!=g_bottleneck_stats->inst_stats_kernel[i].end(); ++it){
                            fprintf(outf, "%d, 0X%x, 0X%x, %d, %d, %d, %d, %d, %d, %d, %s, %s\n", it->second.cache_block_id, it->first, it->second.pc2, it->second.access_cnt, it->second.hit_cnt, it->second.miss_cnt, it->second.rf_cnt, it->second.access_size, it->second.ptx_inst_size_1, it->second.ptx_inst_size_2, it->second.ptx_inst_1.c_str(), it->second.ptx_inst_2.c_str());
                        }
                    
                    }
                
                //}

}

void g_get_stdout_filename(char *filename) {
  char proclnk[1024];
  int fno = fileno(stdout);
  sprintf(proclnk, "/proc/self/fd/%d", fno);
  ssize_t sz = readlink(proclnk, filename, 1024);
  assert(sz >= 0);
  filename[sz] = '\0';
}

void init_con_kernel_opt() {
  if (strcmp(g_con_kernel_mode_opt, "disable") == 0) {
    g_con_kernel_mode = CON_KERNEL_MODE_DISABLE;
  } else if (strcmp(g_con_kernel_mode_opt, "manual") == 0) {
    g_con_kernel_mode = CON_KERNEL_MODE_MANUAL;
  } else {
    assert(0);
  }
  
  sscanf(g_con_kernel_max_tbs_opt, "%d:%d:%d:%d", &g_con_kernel_max_tbs[0], &g_con_kernel_max_tbs[1], &g_con_kernel_max_tbs[2], &g_con_kernel_max_tbs[3]);
  print_con_kernel_opt(stdout);
}

void print_con_kernel_opt(FILE *fout) {

  fprintf(fout, "######################## con_kernel options ########################\n");
  fprintf(fout, "con_kernel_mode %u\n", g_con_kernel_mode);
  fprintf(fout, "num_con_kernels %u\n", g_num_con_kernels);
  fprintf(fout, "con_kernel_max_tbs %s\n", g_con_kernel_max_tbs_opt);
  fprintf(fout, "####################################################################\n");

}

unsigned bottleneck_stats::get_single_max_tbs(unsigned kid) const {
  unsigned tbs[MAX_CON_KERNELS] = {0};
  unsigned result = 8;
  for (unsigned ntb = 8; ntb <= 16; ntb++) {
    tbs[kid] = ntb;
    if (g_the_gpu->m_shader_config->occupancy_check_con_kernel(g_con_kernels, tbs, false)) {
      result = ntb;
    } else {
      break;
    }
  }
  return result;
}

double bottleneck_stats::get_relative_util(unsigned kid, double util) const {
  unsigned single_max_tbs = get_single_max_tbs(kid);
  double expected = (util / g_con_kernel_max_tbs[kid]) * (double)single_max_tbs;
  expected = expected < 1.0 ? expected : 1.0;
  return util / expected;
  
}

void bottleneck_stats::searching_next_tbs(unsigned tbs[]) const {
  unsigned n_sm = g_the_gpu->m_config.num_shader();
  unsigned n_schedulers = g_the_gpu->m_shader_config->gpgpu_num_sched_per_core;
  unsigned n_sub_mem = g_the_gpu->m_memory_config->m_n_mem_sub_partition;
  unsigned n_mem = g_the_gpu->m_memory_config->m_n_mem;

  // scheduler, icnt, dram
  double kernel_vec[MAX_CON_KERNELS][3] = {0.0};
  double sum_vec[3] = {0.0};
  std::vector<unsigned> cres_kernels[3];
  std::vector<unsigned> priority;

  for (unsigned k = 0; k < g_num_con_kernels; k++) {
    double schd = (double)scheduler_util[k] / (n_sm*core_cycles*n_schedulers*1.0);
    double icnt = (double)icnt_m2s_injected_flits[k] / (icnt_cycles*n_sub_mem*.6);
    double dram = (double)dram_bwutil[k] / (n_mem*dram_n_cmd*.8);
    sum_vec[0] += schd;
    sum_vec[1] += icnt;
    sum_vec[2] += dram;
    kernel_vec[k][0] = schd;
    kernel_vec[k][1] = icnt;
    kernel_vec[k][2] = dram;
    unsigned critical_res = max_index3(kernel_vec[k]);
    cres_kernels[critical_res].push_back(k);
    tbs[k] = g_con_kernel_max_tbs[k];
    printf("kid %u, cres %u\n", k, critical_res);
  }

  unsigned hres = max_index3(sum_vec);
  unsigned lres = min_index3(sum_vec);
  unsigned mres = 3 - lres - hres;
  if (cres_kernels[lres].empty() && !cres_kernels[mres].empty()) {
    unsigned tmp = mres;
    mres = lres;
    lres = tmp;
  }
  assert(hres <= 2 && lres <= 2 && mres <= 2);

  // Scenario 1: all utilizations are low
  if (sum_vec[0] < .6 && sum_vec[1] < .5 && sum_vec[2] < .5) {
    for (unsigned i = 0; i < g_num_con_kernels; i++) {
      priority.push_back(i);
    }
    for (unsigned i = 1; i < g_num_con_kernels; i++) {
      for (unsigned j = 0; j < g_num_con_kernels-i; j++) {
	if (g_con_kernel_max_tbs[j] > g_con_kernel_max_tbs[j+1]) {
	  unsigned tmp = priority[j];
	  priority[j] = priority[j+1];
	  priority[j+1] = tmp;
	}
      }
    }
  }
  // Scenario 2, high utilization
  else {

    // If the difference is small, make scheduler as the highest priority
    if (sum_vec[hres]-sum_vec[lres] < .05) {
      lres = 0;
      mres = 1;
      hres = 2;
    }

    assert(cres_kernels[lres].size() <= 2);
    assert(cres_kernels[mres].size() <= 2);
    assert(cres_kernels[hres].size() <= 2);
    if (cres_kernels[lres].size() == 1) {
      priority.push_back(cres_kernels[lres][0]);
    } else if (cres_kernels[lres].size() == 2) {
      unsigned k0 = cres_kernels[lres][0];
      unsigned k1 = cres_kernels[lres][1];
      double r0 = get_relative_util(k0, kernel_vec[k0][lres]);
      double r1 = get_relative_util(k1, kernel_vec[k1][lres]);
      priority.push_back(r0 < r1 ? k0 : k1);
      priority.push_back(r0 < r1 ? k1 : k0);
    }

    if (cres_kernels[mres].size() == 1) {
      priority.push_back(cres_kernels[mres][0]);
    } else if (cres_kernels[mres].size() == 2) {
      unsigned k0 = cres_kernels[mres][0];
      unsigned k1 = cres_kernels[mres][1];
      double r0 = kernel_vec[k0][lres] / kernel_vec[k0][hres];
      double r1 = kernel_vec[k1][lres] / kernel_vec[k1][hres];
      priority.push_back(r0 > r1 ? k0 : k1);
      priority.push_back(r0 > r1 ? k1 : k0);
    }

    if (cres_kernels[hres].size() == 1) {
      priority.push_back(cres_kernels[hres][0]);
    } else if (cres_kernels[hres].size() == 2) {
      unsigned k0 = cres_kernels[hres][0];
      unsigned k1 = cres_kernels[hres][1];
      double r0 = get_relative_util(k0, kernel_vec[k0][hres]);
      double r1 = get_relative_util(k1, kernel_vec[k1][hres]);
      priority.push_back(r0 < r1 ? k0 : k1);
      priority.push_back(r0 < r1 ? k1 : k0);
    }
  }

  printf("Lres kernels: ");
  for (unsigned i = 0; i < cres_kernels[lres].size(); i++) {
    unsigned kid = cres_kernels[lres][i];
    printf("%u ", kid);
  }
  printf("\n");
  printf("Mres kernels: ");
  for (unsigned i = 0; i < cres_kernels[mres].size(); i++) {
    unsigned kid = cres_kernels[mres][i];
    printf("%u ", kid);
  }
  printf("\n");
  printf("Hres kernels: ");
  for (unsigned i = 0; i < cres_kernels[hres].size(); i++) {
    unsigned kid = cres_kernels[hres][i];
    printf("%u ", kid);
  }
  printf("\n");

  unsigned nhres = cres_kernels[hres].size();
  bool issued = false;
  assert(priority.size() == g_num_con_kernels);
  for (unsigned i = 0; i < g_num_con_kernels; i++) {
    unsigned kid = priority[i];
    if (i >= g_num_con_kernels-nhres && hres != 0) {
      if ((sum_vec[hres]-sum_vec[lres] > .2 && sum_vec[hres] > .9)
	  || (sum_vec[hres]-sum_vec[lres] > .4 && sum_vec[hres] > .8 && !cres_kernels[0].empty()))
	break;
    }
    tbs[kid]++;
    if (g_the_gpu->m_shader_config->occupancy_check_con_kernel(g_con_kernels, tbs, false)) {
      issued = true;
      break;
    }
    tbs[kid]--;
  }

  /*
  if (sum_vec[1] > .7 || sum_vec[2] > .7) {
    FILE *lpfile = fopen("_ud_last_perf.tmp", "r");
    if (lpfile) {
      unsigned long long ln[MAX_CON_KERNELS];
      fscanf(lpfile, "%llu %llu %llu %llu", &ln[0], &ln[1], &ln[2], &ln[3]);
      double rstp = .0;
      for (unsigned k = 0; k < g_num_con_kernels; k++) {
	assert(ln[k]);
	double r = (double)thread_insn[k] / (double)ln[k];
	rstp += r;
      }
      if (rstp < (double)g_num_con_kernels) {
	issued = false;
      }
    }
  }


  FILE *lpfile = fopen("_ud_last_perf.tmp", "w");
  assert(lpfile);
  fprintf(lpfile, "%llu %llu %llu %llu\n", thread_insn[0], thread_insn[1], thread_insn[2], thread_insn[3]);
  fclose(lpfile);
  */

  // no available kernel, stop
  if (!issued) {
    FILE *conf = fopen("ud.out", "w");
    assert(conf);
    fprintf(conf, "%u:%u:%u:%u\n", tbs[0], tbs[1], tbs[2], tbs[3]);
    fclose(conf);
    tbs[0] = 88;
  }

  return;

}

