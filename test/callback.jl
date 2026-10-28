@testset "DefaultLobpcgCallback runs" begin
    A = randn(40, 40)
    A = A + A' + 20I
    X = randn(40, 3)

    print_callback = DefaultLobpcgCallback()
    function callback(info)
        _, _, _, _, _, λs = info
        @test λs === info.λs
        @test size(info.X) == size(X)
        @test info.AX ≈ A * info.X atol=1e-12
        @test info.BX ≈ info.X atol=1e-12
        actual = norm.(eachcol(A * info.X - info.X * Diagonal(info.λs)))
        @test info.residual_norms ≈ actual atol=1e-12
        print_callback(info)
    end

    # Run the callback to make sure it works, but don't print anything
    res = redirect_stdout(devnull) do
        lobpcg(A, X, I, Diagonal(A), 1e-6, 100; callback)
    end
    @test maximum(res.residual_norms) < 1e-5
end
