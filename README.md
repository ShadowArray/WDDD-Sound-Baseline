# WDDD-Sound-Baseline
GPGPUsim with a sound baseline configuration.

The configuraiton file (gpgpusim.config) for a Maxwell-like architecutere is under gpgpusim.SoundBaseline/configs/Maxwell-like, where:
  BXOR(Bitwise  XOR) is used for cache set indexing;
  allocate-on-fill is used for cache line allocation at the L1 D-cache;
  128 MSHRs are deployed;
  Xor is used for memory partition mapping.

