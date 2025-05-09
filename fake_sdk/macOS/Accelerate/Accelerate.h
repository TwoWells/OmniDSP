// fake_sdk/macOS/Accelerate/Accelerate.h

#ifndef FAKE_ACCELERATE_FRAMEWORK_ACCELERATE_H  // Unique include guard
#define FAKE_ACCELERATE_FRAMEWORK_ACCELERATE_H

/*
 * This is a fake Accelerate.h header file for IntelliSense purposes on
 * non-Apple platforms.
 *
 * Its primary role will be to include other fake headers from the Accelerate
 * framework's sub-components (like vecLib for vDSP and cblas) that your project
 * uses.
 *
 * Start by identifying which specific Accelerate sub-headers your code includes
 * (e.g., <Accelerate/vecLib/vDSP.h>, <Accelerate/vecLib/cblas.h>)
 * and then create fake versions of those, including them here.
 */

// Include the fake vDSP header.
// This assumes that your include paths are set up so that "Accelerate" is a
// root, and then "vecLib/vDSP.h" can be found within it.
#include <Accelerate/vecLib/vDSP.h>

// If your code uses cblas, you would eventually add:
// #include <Accelerate/vecLib/cblas.h> // You'll need to create
// fake_sdk/macOS/Accelerate/vecLib/cblas.h

#endif  // FAKE_ACCELERATE_FRAMEWORK_ACCELERATE_H
