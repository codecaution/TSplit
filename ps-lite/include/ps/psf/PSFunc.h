#ifndef PSF_
#define PSF_

enum PsfType {
  /* unary ops */
  DensePush,
  DensePull,
  SparsePush,
  SparsePull,
  Nnz,
  Norm2,
  /* Matrix ops */
  PushCols,
  PullCols,
  /* binary ops */
  AddTo,
  Minus,
  Dot,
  Axpy,
  PushPull,
  InitAllZeros,
  Other
};

#endif
