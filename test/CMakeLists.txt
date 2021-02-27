
include_directories(..)
add_executable( test_pyramid
    test_pyramid.cpp )
target_link_libraries( test_pyramid affineklt )

add_executable( test_affine
    test_affine.cpp )
target_link_libraries( test_affine affineklt )

add_executable( test_klt
    test_klt.cpp )
target_link_libraries( test_klt affineklt )

add_executable( test_affineklt
    test_affineklt.cpp )
target_link_libraries( test_affineklt affineklt )
