#ifndef CON_KERNEL_H
#define CON_KERNEL_H

#include <stdio.h>
#include <assert.h>
#include "../abstract_hardware_model.h"

// Zhen: for concurent kernels

// Options
extern char *g_con_kernel_mode_opt;
extern char *g_con_kernel_max_tbs_opt;
extern char *g_con_kernel_l1d_max_set_idx_opt;
extern char *g_con_kernel_l2_max_set_idx_opt;

#define CON_KERNEL_MODE_DISABLE 0
#define CON_KERNEL_MODE_EVEN    1
#define CON_KERNEL_MODE_FULL    2
#define CON_KERNEL_MODE_MANUAL  3
extern unsigned g_con_kernel_mode;
extern unsigned g_num_con_kernels;
#define MAX_CON_KERNELS 4
extern unsigned g_con_kernel_max_tbs[];
extern unsigned g_con_kernel_max_warp_id[];
extern kernel_info_t *g_con_kernels[MAX_CON_KERNELS];
extern unsigned g_con_kernel_l1d_max_set_idx[];
extern unsigned g_con_kernel_l2_max_set_idx[];

void calculate_smk_quota();
void init_con_kernel_opt();
void print_con_kernel_opt(FILE *fout);

inline unsigned WID2KID(unsigned wid) {
  if (g_con_kernel_mode == CON_KERNEL_MODE_DISABLE)
    return 0;
  for (unsigned i = 0; i < g_num_con_kernels; i++) {
    if (wid < g_con_kernel_max_warp_id[i])
      return i;
  }
  assert(0);
}

inline unsigned SID2KID(unsigned sid) {
  if (g_con_kernel_mode == CON_KERNEL_MODE_DISABLE)
    return 0;
  unsigned n_SMs_per_kernel = 16/g_num_con_kernels;
  for (unsigned i = 0; i < g_num_con_kernels; i++) {
    unsigned begin_SM_id = n_SMs_per_kernel*i;
    unsigned end_SM_id = n_SMs_per_kernel*(i+1) - 1;
    if (sid>=begin_SM_id && sid<=end_SM_id){
        //printf("sid=%d, n_SMs_per_kernel=%d, begin_SM_id=%d, end_SM_id=%d -> kid=%d\n", sid, n_SMs_per_kernel, begin_SM_id, end_SM_id, i);
        return i;
    }
  }
  //printf("sid=%d, n_SMs_per_kernel=%d\n", sid, n_SMs_per_kernel);
  return 4;
  assert(0);
}


#endif
