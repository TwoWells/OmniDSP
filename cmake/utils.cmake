# cmake/utils.cmake
# =================
# Contains utility functions for the OmniDSP CMake build system.
# =================

# Function to print a message followed by an indented list, each line prefixed.
# Handles empty lists by printing "(None)".
#
# Usage:
#   print_prefixed_list(<Mode> <Prefix> <MessageString> [<Item1> <Item2> ...])
#
# Arguments:
#   Mode:           The message mode (e.g., STATUS, WARNING, FATAL_ERROR).
#   Prefix:         String to prepend to each output line (e.g., "  ", "    ").
#   MessageString:  The introductory message line (will be prefixed).
#   Item1...:       Optional list of items to print, indented relative to the prefix.
#
# Example:
#   set(my_sources file1.cpp file2.cpp)
#   print_prefixed_list(STATUS "    " "Sources found:" ${my_sources})
#   set(empty_list "")
#   print_prefixed_list(STATUS "    " "Empty list:" ${empty_list}) # Prints (None)
#
function(print_prefixed_list MODE PREFIX MESSAGE_STRING)
    # Print the introductory message with the prefix
    message(${MODE} "${PREFIX}${MESSAGE_STRING}")

    # ARGN contains all arguments *after* the explicitly named ones (MESSAGE_STRING)
    if(ARGN)
        # If the list is NOT empty, loop through and print items
        foreach(item ${ARGN})
            # Print each item, indented with a bullet point relative to the prefix
            message(${MODE} "${PREFIX}  - ${item}")
        endforeach()
    else()
        # If the list IS empty, print (None) indented
        message(${MODE} "${PREFIX}  (None)")
    endif()
endfunction()

message(STATUS "Loaded CMake utility functions from cmake/utils.cmake")
