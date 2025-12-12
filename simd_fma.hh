#include <Eigen/Dense>
#include <cmath>
#include <type_traits>

// Scalar case (e.g., for `float` or `double`)
template<typename T>
typename std::enable_if<std::is_arithmetic<T>::value, T>::type
simd_fma(T x, T y, T z) { return std::fma(x, y, z); }

#define NAIVE_LOOP_IMPLEMENTATION 0

template<typename Traits, bool NeedsHalfPacket> struct PacketTypeImpl;
template<typename Traits> struct PacketTypeImpl<Traits, /* NeedsHalfPacket = */ 0> { using type = typename Traits::type; static constexpr size_t size = Traits::size; };
template<typename Traits> struct PacketTypeImpl<Traits, /* NeedsHalfPacket = */ 1> { using type = typename Traits::half; static constexpr size_t size = Traits::size / 2; static_assert(Traits::HasHalfPacket, "Half-packet type not available"); };

template<typename Scalar, size_t VEC_WIDTH>
struct PacketType {
    using Traits = Eigen::internal::packet_traits<Scalar>;
    static constexpr bool NeedsHalfPacket = (VEC_WIDTH == Traits::size / 2);
    using Impl = PacketTypeImpl<Traits, NeedsHalfPacket>;
    using type = typename Impl::type;
    static constexpr size_t size = Impl::size;
};

#include <iostream>

// Overload for Eigen types.
template<typename T>
typename std::enable_if<std::is_base_of<Eigen::DenseBase<T>, T>::value, T>::type
simd_fma(T x, T y, T z) {
#if NAIVE_LOOP_IMPLEMENTATION
    T result;
    for (int i = 0; i < x.size(); ++i) {
        result[i] = std::fma(x[i], y[i], z[i]);
    }
#else
    static_assert((T::RowsAtCompileTime == 1) || (T::ColsAtCompileTime == 1), "Expected a 1D array");
    static constexpr int VEC_WIDTH = (T::ColsAtCompileTime == 1) ? T::RowsAtCompileTime : T::ColsAtCompileTime;

    using PT = PacketType<typename T::Scalar, VEC_WIDTH>;
    using Packet = typename PT::type;
    static constexpr size_t packet_size = PT::size;
    static constexpr size_t Aligned = Eigen::Aligned;

    // std::cout << "VEC_WIDTH: " << VEC_WIDTH << std::endl;
    // std::cout << "packet_size: " << packet_size << std::endl;

    static_assert(VEC_WIDTH % packet_size == 0, "Vector width should be divisible by the packet size.");

    // Force a SIMD+FMA implementation
    // (still requires -mfma or -march=native on x86; otherwise `pmadd` will
    // be broken into calls to `pmul` and `padd`).
    T result;
    for (size_t i = 0; i < VEC_WIDTH; i += packet_size) {
        Eigen::internal::pstoret<typename T::Scalar, Packet, Aligned>(result.data() + i,
            Eigen::internal::pmadd(Eigen::internal::ploadt_ro<Packet, Aligned>(x.data() + i),
                                   Eigen::internal::ploadt_ro<Packet, Aligned>(y.data() + i),
                                   Eigen::internal::ploadt_ro<Packet, Aligned>(z.data() + i))
        );
    }
#endif

    return result;
}