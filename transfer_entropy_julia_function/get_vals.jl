import CoTETE
using DataFrames
using DelimitedFiles

function read_times(filename)
    times=Dict{Int,Vector{Float64}}()
    n=0
    for line in readlines(filename)
        values = split(line,',')
        values = parse.(Float64,values)

        times[Int(values[1])+1]= values[2:end]
        n+=1
    end

    return times, n
end

# function to check file existance
println("in julia...")
parameters = CoTETE.CoTETEParameters(l_x = 1, l_y = 1,  use_exclusion_windows=false);


function wait_on_file(filepath_name,seconds)
    while !isfile(filepath_name)
        sleep(seconds)
    end
    return 0
end

global iter=0

while true

    println("julia: waiting for files...")

    wait_on_file("files/python_flag.txt",20)

    println("julia: iteration $iter")
    global iter+=1
    # load in times

    
    times, n  = read_times("files/times.csv")

    # do estimates
    # println("julia: processing")
    save_vals = zeros(Float64, n,n)
    for i in 1:n
        for j in 1:n
            # print(i,j)
            source = times[i]
            target = times[j]
            if (length(source) ==1) | (length(target)==1)
                save_vals[i,j] = NaN
            else
                save_vals[i,j] = CoTETE.estimate_TE_from_event_times(parameters, target, source)
            end
        end
    end
    # println("julia: saving")
    # write res
    writedlm( "files/julia_estimates.csv",  save_vals, ',')

    # write flag file
    writedlm( "files/julia_flag.txt",  n, ',')


    # delete file
    rm("files/python_flag.txt")



    # isfile("files/python_done.txt") || break
end

println("julia: ended.")

