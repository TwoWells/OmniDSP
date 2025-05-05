#include "utils.hpp"

#include <ippcore.h>

#include <iostream>
#include <string>

namespace OmniDSP::intelipp::utils {

  // Definition (no inline keyword here)
  Status ipp_status_to_omnidsp_status(IppStatus status)
  {
    if (status == ippStsNoErr) {
      return Status::Success;
    }
    std::cerr << "IPP Error: " << ippGetStatusString(status)
              << " (Code: " << status << ")" << std::endl;

    if (status == ippStsNullPtrErr || status == ippStsSizeErr
        || status == ippStsStepErr || status == ippStsBadArgErr
        || status == ippStsOutOfRangeErr || status == ippStsFftOrderErr
        || status == ippStsFftFlagErr) {
      return Status::InvalidArgument;
    }
    if (status == ippStsMemAllocErr) {
      return Status::AllocationError;
    }
    if (status == ippStsContextMatchErr) {
      return Status::InvalidOperation;  // Or potentially InvalidArgument?
    }
    // ... rest of the mapping ...
    return Status::BackendError;
  }

  // ... (definitions of other helpers if moved from header) ...

}  // namespace OmniDSP::intelipp::utils
