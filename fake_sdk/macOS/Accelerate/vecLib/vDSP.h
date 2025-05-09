// fake_sdk/macOS/Accelerate/vecLib/vDSP.h

#ifndef FAKE_ACCELERATE_VECLIB_VDSP_H  // Unique include guard
#define FAKE_ACCELERATE_VECLIB_VDSP_H

// It's good practice to include standard headers for types like size_t,
// but vDSP_Length is specifically unsigned long as per your request.
// #include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// --- Basic Types ---

/**
 * @typedef vDSP_Length
 * @brief An unsigned-integer value that represents the size of vectors and the
 * indices of elements in vectors.
 * @availability iOS, iPadOS, Mac Catalyst, macOS, tvOS, visionOS, watchOS
 */
typedef unsigned long vDSP_Length;

/**
 * @typedef vDSP_Stride
 * @brief An integer value that represents the differences between indices of
 * elements, including the lengths of strides.
 * @availability iOS, iPadOS, Mac Catalyst, macOS, tvOS, visionOS, watchOS
 */
typedef long vDSP_Stride;

// --- Complex Number Structures ---

/**
 * @struct DSPComplex
 * @brief A structure that represents a single-precision complex value.
 * @availability iOS, iPadOS, Mac Catalyst, macOS, tvOS, visionOS, watchOS
 *
 * @discussion Complex data are stored as ordered pairs of floating-point
 * numbers. Because they are stored as ordered pairs, complex vectors require
 * address strides that are multiples of two.
 */
typedef struct DSPComplex {
  float real; /**< The real part of the value. */
  float imag; /**< The imaginary part of the value. */
} DSPComplex;

/**
 * @struct DSPDoubleComplex
 * @brief A structure that represents a double-precision complex value.
 * @availability iOS, iPadOS, Mac Catalyst, macOS, tvOS, visionOS, watchOS
 *
 * @discussion Double complex data are stored as ordered pairs of
 * double-precision floating-point numbers. Because they are stored as ordered
 * pairs, complex vectors require address strides that are multiples of two.
 */
typedef struct DSPDoubleComplex {
  double real; /**< The real part of the value. */
  double imag; /**< The imaginary part of the value. */
} DSPDoubleComplex;

// --- Opaque Types ---
// For vDSP_DFT_Interleaved_Setup (single-precision), we define it as an opaque
// pointer. This is common for setup objects where the internal structure is not
// public.
typedef struct OpaquevDSPDFTInterleavedSetup* vDSP_DFT_Interleaved_Setup;

// For vDSP_DFT_Interleaved_SetupD (double-precision), we define it as an opaque
// pointer.
typedef struct OpaquevDSPDFTInterleavedSetupD* vDSP_DFT_Interleaved_SetupD;
// An alternative simpler way, if preferred for both:
// typedef void* vDSP_DFT_Interleaved_Setup;
// typedef void* vDSP_DFT_Interleaved_SetupD;

// --- Enumerations ---

/**
 * @enum vDSP_DFT_Direction
 * @brief An enumeration that specifies whether to perform a forward or inverse
 * DFT.
 * @availability iOS, iPadOS, Mac Catalyst, macOS, tvOS, visionOS, watchOS
 */
typedef enum {
  vDSP_DFT_FORWARD = 1, /**< A constant that specifies a forward transform. */
  vDSP_DFT_INVERSE = -1 /**< A constant that specifies an inverse transform. */
} vDSP_DFT_Direction;

/**
 * @enum vDSP_DFT_RealtoComplex
 * @brief An enumeration that specifies the transform type for DFT operations.
 * @availability iOS, iPadOS, Mac Catalyst, macOS, tvOS, visionOS, watchOS
 * @discussion While the original API might suggest a `bool` underlying type,
 * standard C enums are integer-based. This definition uses integers
 * to represent the distinct transform types.
 */
typedef enum {
  /** @brief Specifies a complex-to-complex transform. */
  vDSP_DFT_Interleaved_ComplextoComplex = 0,
  /** @brief Specifies a real-to-complex transform. */
  vDSP_DFT_Interleaved_RealtoComplex = 1
} vDSP_DFT_RealtoComplex;

// --- Function Declarations ---

/**
 * @brief Returns a setup structure that contains precalculated data for forward
 * and inverse, single-precision interleaved discrete Fourier transform (DFT)
 * functions.
 * @availability iOS 15.0+, iPadOS 15.0+, Mac Catalyst 15.0+, macOS 12.0+,
 * tvOS 15.0+, visionOS 1.0+, watchOS 8.0+
 *
 * @param Previous An existing vDSP_DFT_Interleaved_Setup structure that shares
 * memory and direction with the setup structure that this function returns.
 * Pass NULL (or 0 for C) to create an object with newly initialized and
 * allocated memory. To ensure correct operation, if you specify a previous
 * setup structure it must share the same direction as the Direction parameter.
 * @param Length For complex-to-complex transforms, the number of complex
 * elements. For real-to-complex transforms, the number of real elements divided
 * by 2.
 * @param Direction A flag that specifies the transform direction.
 * Pass vDSP_DFT_FORWARD to transform from the time domain to the frequency
 * domain. Pass vDSP_DFT_INVERSE to transform from the frequency domain to the
 * time domain.
 * @param RealtoComplex A flag that specifies the transform type.
 * To transform from complex to complex, pass
 * vDSP_DFT_Interleaved_ComplextoComplex. To transform from real to complex,
 * pass vDSP_DFT_Interleaved_RealtoComplex.
 * @return Returns a vDSP_DFT_Interleaved_Setup object, or NULL (0) if the
 * function fails, either from insufficient memory or because Length doesn’t
 * satisfy the requirements.
 */
vDSP_DFT_Interleaved_Setup vDSP_DFT_Interleaved_CreateSetup(
    vDSP_DFT_Interleaved_Setup Previous,
    vDSP_Length Length,
    vDSP_DFT_Direction Direction,
    vDSP_DFT_RealtoComplex RealtoComplex);

/**
 * @brief Returns a setup structure that contains precalculated data for forward
 * and inverse, double-precision interleaved discrete Fourier transform (DFT)
 * functions.
 * @availability iOS 15.0+, iPadOS 15.0+, Mac Catalyst 15.0+, macOS 12.0+,
 * tvOS 15.0+, visionOS 1.0+, watchOS 8.0+
 *
 * @param Previous An existing vDSP_DFT_Interleaved_SetupD structure that shares
 * memory and direction with the setup structure that this function returns.
 * Pass NULL (or 0 for C) to create an object with newly initialized and
 * allocated memory. To ensure correct operation, if you specify a previous
 * setup structure it must share the same direction as the Direction parameter.
 * @param Length For complex-to-complex transforms, the number of complex
 * elements. For real-to-complex transforms, the number of real elements divided
 * by 2.
 * @param Direction A flag that specifies the transform direction.
 * Pass vDSP_DFT_FORWARD to transform from the time domain to the frequency
 * domain. Pass vDSP_DFT_INVERSE to transform from the frequency domain to the
 * time domain.
 * @param RealtoComplex A flag that specifies the transform type.
 * To transform from complex to complex, pass
 * vDSP_DFT_Interleaved_ComplextoComplex. To transform from real to complex,
 * pass vDSP_DFT_Interleaved_RealtoComplex.
 * @return Returns a vDSP_DFT_Interleaved_SetupD object, or NULL (0) if the
 * function fails, either from insufficient memory or because Length doesn’t
 * satisfy the requirements.
 */
vDSP_DFT_Interleaved_SetupD vDSP_DFT_Interleaved_CreateSetupD(
    vDSP_DFT_Interleaved_SetupD Previous,
    vDSP_Length Length,
    vDSP_DFT_Direction Direction,
    vDSP_DFT_RealtoComplex RealtoComplex);

/**
 * @brief Calculates the single-precision discrete Fourier transform (DFT) for a
 * vector of interleaved complex values.
 * @availability iOS 15.0+, iPadOS 15.0+, Mac Catalyst 15.0+, macOS 12.0+,
 * tvOS 15.0+, visionOS 1.0+, watchOS 8.0+
 *
 * @param Setup The DFT setup structure for this transform. This structure is
 * not modified by the function.
 * @param Iri A single-precision vector that contains the input values. This
 * vector is not modified by the function.
 * @param Ori A single-precision vector that contains the output values.
 * The output can equal the input, but this function doesn’t support any other
 * overlap of the input and output vectors.
 *
 * @discussion This function supports in-place operation where the Ori and Iri
 * parameters point to the same memory. The transform length must equal the
 * transform length specified in the setup structure.
 * @important For best performance, make sure the two vector addresses you pass
 * to this function are 16-byte-aligned.
 */
void vDSP_DFT_Interleaved_Execute(
    vDSP_DFT_Interleaved_Setup const Setup,
    const DSPComplex* Iri,
    DSPComplex* Ori);

/**
 * @brief Calculates the double-precision discrete Fourier transform (DFT) for a
 * vector of interleaved complex values.
 * @availability iOS 15.0+, iPadOS 15.0+, Mac Catalyst 15.0+, macOS 12.0+,
 * tvOS 15.0+, visionOS 1.0+, watchOS 8.0+
 *
 * @param Setup The DFT setup structure for this transform. This structure is
 * not modified by the function.
 * @param Iri A double-precision vector that contains the input values. This
 * vector is not modified by the function.
 * @param Ori A double-precision vector that contains the output values.
 * The output can equal the input, but this function doesn’t support any other
 * overlap of the input and output vectors.
 *
 * @discussion This function supports in-place operation where the Ori and Iri
 * parameters point to the same memory. The transform length must equal the
 * transform length specified in the setup structure.
 * @important For best performance, make sure the two vector addresses you pass
 * to this function are 16-byte-aligned.
 */
void vDSP_DFT_Interleaved_ExecuteD(
    vDSP_DFT_Interleaved_SetupD const Setup,
    const DSPDoubleComplex* Iri,
    DSPDoubleComplex* Ori);

/**
 * @brief Releases a single-precision discrete Fourier transform (DFT) setup
 * structure.
 * @availability iOS 15.0+, iPadOS 15.0+, Mac Catalyst 15.0+, macOS 12.0+,
 * tvOS 15.0+, visionOS 1.0+, watchOS 8.0+
 *
 * @param Setup The setup structure to destroy.
 *
 * @discussion
 * @important This function isn’t fully thread-safe. Don’t call this function
 * concurrently with any function that uses or shares its underlying storage
 * with the setup structure.
 */
void vDSP_DFT_Interleaved_DestroySetup(vDSP_DFT_Interleaved_Setup Setup);

/**
 * @brief Releases a double-precision discrete Fourier transform (DFT) setup
 * structure.
 * @availability iOS 15.0+, iPadOS 15.0+, Mac Catalyst 15.0+, macOS 12.0+,
 * tvOS 15.0+, visionOS 1.0+, watchOS 8.0+
 *
 * @param Setup The setup structure to destroy.
 *
 * @discussion
 * @important This function isn’t fully thread-safe. Don’t call this function
 * concurrently with any function that uses or shares its underlying storage
 * with the setup structure.
 */
void vDSP_DFT_Interleaved_DestroySetupD(vDSP_DFT_Interleaved_SetupD Setup);

/*
 * Add other vDSP types
 * and function declarations from <Accelerate/vecLib/vDSP.h>
 * that your OmniDSP project uses.
 */

#ifdef __cplusplus
}
#endif

#endif  // FAKE_ACCELERATE_VECLIB_VDSP_H
