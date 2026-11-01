if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
  message(FATAL_ERROR "This project currently requires native Apple Silicon")
endif()

include(FetchContent)
find_package(Git REQUIRED)
find_program(MOE_PATCH_EXECUTABLE patch REQUIRED)
set(GGML_METAL ON CACHE BOOL "" FORCE)
set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)
set(GGML_CCACHE OFF CACHE BOOL "" FORCE)
set(GGML_BLAS ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_USE_PREBUILT_UI OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMIT "555d1ec" CACHE STRING "" FORCE)
set(LLAMA_BUILD_NUMBER 1 CACHE STRING "" FORCE)
FetchContent_Declare(moe_llama
  URL https://codeload.github.com/memoriaru/llama.cpp/tar.gz/555d1ec9daa4564cb06adb39c469acfdcd1d1e93
  URL_HASH SHA256=774d5c5c88a7b4b6122478fe4fb5e19d950c626c64c11a670c1a997a0e4ee65d
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  PATCH_COMMAND ${MOE_PATCH_EXECUTABLE} -p1 -i ${CMAKE_CURRENT_SOURCE_DIR}/llama-metal.patch)
FetchContent_MakeAvailable(moe_llama)

add_executable(moe-profile-model EXCLUDE_FROM_ALL profile-model.cpp)
target_link_libraries(moe-profile-model PRIVATE llama ggml)
target_compile_features(moe-profile-model PRIVATE cxx_std_17)

add_executable(moe-quality-logits EXCLUDE_FROM_ALL quality-logits.cpp)
target_link_libraries(moe-quality-logits PRIVATE llama ggml)
target_compile_features(moe-quality-logits PRIVATE cxx_std_17)

add_test(NAME moe-file-experts COMMAND $<TARGET_FILE:test-expert-pool> --file)
add_custom_target(moe-mac-check
  COMMAND $<TARGET_FILE:test-batch-alloc>
  COMMAND $<TARGET_FILE:test-expert-pool>
  COMMAND $<TARGET_FILE:test-expert-pool> --file
  DEPENDS test-batch-alloc test-expert-pool
  USES_TERMINAL)
