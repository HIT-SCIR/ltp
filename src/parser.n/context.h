#ifndef __LTP_PARSERN_CONTEXT_H__
#define __LTP_PARSERN_CONTEXT_H__

namespace ltp {
namespace depparser {

struct Context {
  int S0, S1, S2, N0, N1, N2;
  int S0L, S0R, S0L2, S0R2, S0LL, S0RR;
  int S1L, S1R, S1L2, S1R2, S1LL, S1RR;
};

} //  namespace depparser
} //  namespace ltp

#endif  //  end for __LTP_PARSERN_CONTEXT_H__
