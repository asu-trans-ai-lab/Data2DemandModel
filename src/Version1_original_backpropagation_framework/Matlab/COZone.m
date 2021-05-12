classdef COZone<handle
    properties
        ozone_id 
        node_id 
        population
        trip_rate
        target_generation
        estimated_generation
        total_gamma
        link_ozone_error_signal
    end    
    methods
        function obj=COZone()% ¹¹Ôìº¯Êý
        end       
        function obj=Ref_Gen(obj,trip_rate,population)
                obj.target_generation=trip_rate*population;
                obj.estimated_generation=trip_rate*population*0;%(unifrnd(0.8,1.2));
        end        
    end   
end

