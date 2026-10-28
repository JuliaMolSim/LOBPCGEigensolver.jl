# Build a well-conditioned Hermitian matrix with a known spectrum
function test_matrix(T, N; shift=20)
    A = rand(T, N, N)
    Hermitian(A + A') + shift * I
end

# Reference: the nev smallest eigenvalues by dense diagonalization
ref_eigvals(A, nev) = sort(eigen(Hermitian(Matrix(A))).values)[1:nev]

@testset "lobpcg matches dense diagonalization" begin
    for T in (Float64, ComplexF64)
        N, nev = 60, 5
        A = test_matrix(T, N)
        X0 = rand(T, N, nev)

        res = lobpcg(A, X0, I, Diagonal(A), 1e-9, 200)

        @test res.converged
        @test res.λ ≈ ref_eigvals(A, nev)
        @test maximum(res.residual_norms) < 1e-8
        # Returned eigenvectors are orthonormal
        @test norm(res.X' * res.X - I) < 1e-8
        # And they actually solve the eigenproblem
        @test norm(A * res.X - res.X * Diagonal(res.λ)) < 1e-7
    end
end

@testset "lobpcg without preconditioner" begin
    N, nev = 60, 4
    A = test_matrix(Float64, N)
    X0 = rand(N, nev)
    res = lobpcg(A, X0, I, I, 1e-8, 200)
    @test res.λ ≈ ref_eigvals(A, nev) atol=1e-7
    @test maximum(res.residual_norms) < 1e-7
end

# A well-conditioned symmetric positive-definite metric with eigenvalues in [1, 3].
function spd_metric(N)
    Q = Matrix(qr(randn(N, N)).Q)
    Hermitian(Q * Diagonal(range(1.0, 3.0, N)) * Q')
end

@testset "generalized eigenproblem (B != I)" begin
    N, nev = 60, 4
    A = test_matrix(Float64, N)
    B = spd_metric(N)
    X0 = rand(N, nev)

    res = lobpcg(A, X0, B, Diagonal(A), 1e-9, 200)
    ref = sort(eigen(Matrix(A), Matrix(B)).values)[1:nev]
    @test res.converged
    @test res.λ ≈ ref atol=1e-7
    @test res.AX ≈ A * res.X atol=1e-12
    @test res.BX ≈ B * res.X atol=1e-12
    @test res.residual_norms ≈ norm.(eachcol(A * res.X - B * res.X * Diagonal(res.λ))) atol=1e-12
    # Residual of the generalized eigenproblem A x = λ B x
    @test norm(A * res.X - B * res.X * Diagonal(res.λ)) < 1e-6
    # B-orthonormality of the returned eigenvectors
    @test norm(res.X' * B * res.X - I) < 1e-7
end

@testset "partial convergence is checked after sorting" begin
    A = Diagonal(vcat(0., 0., 1.:18.))
    tol = 1e-6
    X = zeros(20, 2)
    # Both target eigenvalues are zero. The second column has the lower Rayleigh
    # quotient but a residual above tol; only the first column can be locked.
    X[1, 1] = 1
    X[3, 1] = tol / 2
    X[2, 2] = 1
    X[20, 2] = 1.5tol / A[20, 20]

    for maxiter in (0, 5)
        res = lobpcg(A, X, I, I, tol, maxiter; miniter=0, n_conv_check=1)
        actual = norm.(eachcol(A * res.X - res.X * Diagonal(res.λ)))
        @test issorted(res.λ)
        @test res.converged == (maxiter > 0)
        @test res.n_iter == (maxiter > 0 ? 1 : 0)
        @test res.residual_norms ≈ actual atol=1e-13
        @test (actual[1] < tol) == res.converged
        # The locked residual must be retained rather than replaced by zero.
        @test all(r -> r ≈ tol / 2, res.residual_history[2, :])
        @test res.residual_norms == res.residual_history[:, end]
        _, _, _, _, _, _, n_matvec = res
        @test n_matvec == res.n_matvec
    end
end
