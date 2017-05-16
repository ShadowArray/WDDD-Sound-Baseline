#ifndef BOTTLENECK_STATS_H
#define BOTTLENECK_STATS_H

#include <map>
#include "con_kernel.h"

struct accessed_inst_info{
    //void ptx_print_insn( address_type pc, FILE *fp )
    unsigned pc2;
    std::string ptx_inst_1;
    unsigned ptx_inst_size_1;
    std::string ptx_inst_2;
    unsigned ptx_inst_size_2;
    unsigned access_size;
    unsigned access_cnt;
    unsigned hit_cnt;
    unsigned miss_cnt;
    unsigned rf_cnt;

    unsigned cache_block_id;
};
//insturction pc -> stats
typedef std::map<unsigned, struct accessed_inst_info > instruction_stats;

struct accessed_cache_block_info{
    unsigned access_cnt;
    unsigned hit_cnt;
    unsigned miss_cnt;
    unsigned rf_cnt;
    float miss_rate;
    float rf_per_access;
};

//cache_block_id -> stats
typedef std::map<unsigned, struct accessed_cache_block_info > cache_block_stats;

struct access_cnt_based_info{
    unsigned block_cnt;
    
    unsigned hit_cnt;
    unsigned miss_cnt;
    unsigned rf_cnt;
    float miss_rate; 
    float rf_per_access;
};

//access_cnt -> stats
typedef std::map<unsigned, struct access_cnt_based_info > access_cnt_based_stats;

// Zhen: stats for bottleneck analysis
class bottleneck_stats {
private:
  //void compute_stats(const unsigned long long *stats, unsigned *maxid, unsigned long long *max, unsigned *nzeros, unsigned long long *avg) const;
  void print_item_vs(FILE *outf, const char* name, const unsigned long long *util, unsigned long long denom) const;
  void print_item_vv(FILE *outf, const char* name, const unsigned long long *util, const unsigned long long *denom) const;
  void print_item_ss(FILE *outf, const char* name, unsigned long long util, unsigned long long denom) const;
  double cal_variance(double util[3]) const;
public:
  //Added on 2017-03-17, for MDB, Hongwen
  unsigned MDB_global_keep_num;
  unsigned long long MDB_sampled_rsfails[16];//reset sampled rsfails
  unsigned long long n_l1d_bypass_num[MAX_CON_KERNELS];
  unsigned long long bypass_l1d_request_latency[MAX_CON_KERNELS];
  unsigned long long access_l1d_request_latency[MAX_CON_KERNELS];
  unsigned long long n_l1d_bp_rsfails[MAX_CON_KERNELS];
  unsigned long long n_l1d_max_bp_in_circle_num[MAX_CON_KERNELS];
  unsigned long long n_l1d_accu_bp_in_circle_num[MAX_CON_KERNELS];
  unsigned long long n_l1d_accu_bp_in_circle_cycles[MAX_CON_KERNELS];

  //Added on 2017-02-11, Hongwen
  unsigned long long Cinst_cnt_kernel[MAX_CON_KERNELS];
  unsigned long long Minst_cnt_kernel[MAX_CON_KERNELS];
  unsigned long long Minst_Smem_cnt_kernel[MAX_CON_KERNELS];
  unsigned long long Minst_Ccache_cnt_kernel[MAX_CON_KERNELS];
  unsigned long long Minst_Tcache_cnt_kernel[MAX_CON_KERNELS];
  unsigned long long Minst_Dcache_cnt_kernel[MAX_CON_KERNELS];
  unsigned long long Minst_other_cnt_kernel[MAX_CON_KERNELS];
  unsigned long long load_inst_kernel[MAX_CON_KERNELS];
  unsigned long long store_inst_kernel[MAX_CON_KERNELS];
  unsigned long long barrier_inst_kernel[MAX_CON_KERNELS];
  //per shader statistics
  unsigned long long Minst_Dcache_cnt_kernel_shader[16][MAX_CON_KERNELS];
  unsigned long long n_l1d_access_kernel_shader[16][MAX_CON_KERNELS];
  unsigned long long n_1ld_mshrs_full;
  unsigned long long n_l1d_missq_full;
  unsigned long long n_l1d_slot_resfail[MAX_CON_KERNELS];
  unsigned long long n_l1d_mshrs_resfail[MAX_CON_KERNELS];
  unsigned long long n_l1d_missq_resfail[MAX_CON_KERNELS];
  
  //for memory instruction limiting
  unsigned long long Inflight_Cinst_accu_kernel_shader[16][MAX_CON_KERNELS];
  unsigned long long Inflight_Minst_accu_kernel_shader[16][MAX_CON_KERNELS];
  unsigned long long Inflight_Cinst_accu_kernel[MAX_CON_KERNELS];
  unsigned long long Inflight_Minst_accu_kernel[MAX_CON_KERNELS];
  unsigned long long Minst_limit_kernel[16][MAX_CON_KERNELS];
  unsigned long long L1D_access_set[32];

  //SMK
  void smk_tbs() const;
  unsigned Winst_quota_kernel[MAX_CON_KERNELS];

  //Added on 2017-02-24, Hongwen
  //record runtime statistics
  void print_con_kernel_runtime_stats(FILE *outf, std::vector<unsigned long long*> runtime_stats) const;
  unsigned sampled_epoch_cycle;
  unsigned long long sampled_Winst_cnt[MAX_CON_KERNELS];
  unsigned long long sampled_Cinst_cnt[MAX_CON_KERNELS];
  unsigned long long sampled_Minst_cnt[MAX_CON_KERNELS];
  unsigned long long sampled_l1d_access_cnt[MAX_CON_KERNELS];
        
  std::vector<unsigned long long*> runtime_Winst_stats;
  std::vector<unsigned long long*> runtime_Cinst_stats;
  std::vector<unsigned long long*> runtime_Minst_stats;
  std::vector<unsigned long long*> runtime_l1d_access_stats;

  //Added on 2017-02-01, Hongwen
  instruction_stats inst_stats_kernel[MAX_CON_KERNELS];
  cache_block_stats cache_block_access_stats_kernel[MAX_CON_KERNELS];
  access_cnt_based_stats access_cnt_based_stats_kernel[MAX_CON_KERNELS];
  void print_icache_stats(FILE *outf, unsigned icache_line_size, unsigned icache_size) const;

  // For optimal TB searching
  void searching_next_tbs(unsigned tbs[]) const;
  void warped_slicer_tbs() const;
  void interleaved_tbs() const;
  double get_relative_util(unsigned kid, double util) const;
  unsigned get_single_max_tbs(unsigned kid) const;

  bottleneck_stats() { clear(); }
  void print(FILE *outf) const;
  //void print_average_only(FILE *outf) const;
  void clear();
  void print_con_kernels(FILE *outf) const;
  void get_kernel_factors(double *advance, double *retreat) const;
  void print_kernel_factors(FILE *outf, double *advance, double *retreat) const;
  void normalize_vector(const double *in, double *out, unsigned size) const;
  double get_advance_factor(const double *nutil, const double *nremain) const;
  unsigned max_index3(const double *array3) const;
  unsigned min_index3(const double *array3) const;
  unsigned find_low_resource(const double *array3, const bool *pcr) const;
  //double get_vector_distance(const double *v1, const double *v2, unsigned size) const;
  //void conkernel_period();
  
  // Context resources
  unsigned long long sm_thread_util[MAX_CON_KERNELS];
  unsigned long long sm_tb_util[MAX_CON_KERNELS];
  unsigned long long sm_reg_util[MAX_CON_KERNELS];
  unsigned long long sm_smem_util[MAX_CON_KERNELS];

  // DRAM BW
  unsigned long long dram_n_cmd;
  unsigned long long dram_bwutil[MAX_CON_KERNELS];
  unsigned long long dram_activity;

  // ICNT
  unsigned long long icnt_cycles;
  unsigned long long icnt_s2m_injected_flits[MAX_CON_KERNELS];
  unsigned long long icnt_m2s_injected_flits[MAX_CON_KERNELS];
  unsigned long long icnt_s2m_total_packets[MAX_CON_KERNELS];
  unsigned long long icnt_m2s_total_packets[MAX_CON_KERNELS];

  // L2
  unsigned long long l2_cycles;
  unsigned long long l2_mshr_util[MAX_CON_KERNELS];
  unsigned long long l2_missq_util[MAX_CON_KERNELS];
  unsigned long long l2_tag_util[MAX_CON_KERNELS];
  unsigned long long l2_data_util[MAX_CON_KERNELS];
  unsigned long long l2_fill_util[MAX_CON_KERNELS];
  unsigned long long n_l2_access[MAX_CON_KERNELS];
  unsigned long long n_l2_hit[MAX_CON_KERNELS];
  unsigned long long n_l2_miss[MAX_CON_KERNELS];
  unsigned long long n_l2_resfail[MAX_CON_KERNELS];

  // L1
  unsigned long long l1_mshr_util[MAX_CON_KERNELS];
  unsigned long long l1_missq_util[MAX_CON_KERNELS];
  unsigned long long n_l1_access[MAX_CON_KERNELS];
  unsigned long long n_l1_hit[MAX_CON_KERNELS];
  unsigned long long n_l1_miss[MAX_CON_KERNELS];
  unsigned long long n_l1_resfail[MAX_CON_KERNELS];
  unsigned long long l1_tag_util[MAX_CON_KERNELS];
  unsigned long long l1_data_util[MAX_CON_KERNELS];
  unsigned long long l1_fill_util[MAX_CON_KERNELS];
  //unsigned long long l1i_tag_util[MAX_CON_KERNELS];

  // Shader core components
  unsigned long long core_cycles;
  unsigned long long thread_insn[MAX_CON_KERNELS];
  unsigned long long scheduler_util[MAX_CON_KERNELS];
  unsigned long long mem_pipe_util[MAX_CON_KERNELS];
  unsigned long long sp_pipe_util[MAX_CON_KERNELS];
  unsigned long long sfu_pipe_util[MAX_CON_KERNELS];
  unsigned long long smem_cycles[MAX_CON_KERNELS];
  //unsigned long long reg_util[MAX_CON_KERNELS];
  //unsigned long long l1_pipe_util[MAX_CON_KERNELS];
  //unsigned long long ldst_memory_cycle[MAX_CON_KERNELS];
  //unsigned long long l1_reservation_fail[MAX_CON_KERNELS];

  //unsigned long long tmp_counter;

  unsigned long long n_gmem_load_insns;
  unsigned long long n_gmem_load_accesses;
  unsigned long long n_gmem_load_useful_bytes;
  unsigned long long n_gmem_load_transaction_bytes;
  unsigned long long n_smem_access_insn;
  unsigned long long n_smem_accesses;

  unsigned long long schd_fetch;
  unsigned long long schd_sync;
  unsigned long long schd_control_hzd;
  unsigned long long schd_data_hzd;
  unsigned long long schd_struct_hzd;
  unsigned long long schd_run;

};

// defined and init in gpu-sim.cc
extern class bottleneck_stats *g_bottleneck_stats;

#endif
