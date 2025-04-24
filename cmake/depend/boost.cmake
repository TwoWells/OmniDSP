# cmake/depend/boost.cmake
# =========================
# Finds the Boost libraries using find_package.
# =========================

message(STATUS "  Finding Boost dependency...")

# Find Boost (Using find_package is recommended over FetchContent for Boost)
find_package(Boost REQUIRED)
message(STATUS "    Found Boost version ${Boost_VERSION}")
