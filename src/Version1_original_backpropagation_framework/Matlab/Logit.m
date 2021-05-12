function [g_path_vector] = Logit(g_od_vector,g_path_vector)
[~,nb_od]=size(g_od_vector);
[~,nb_path]=size(g_path_vector);

% using softmax function
% path_estimated_proportion =
% exp(-VOT*path_travel_time-path_cost)/sum(exp(-VOT*path_travel_time-path_cost),for all path)

%%== Step: A logit dominator function is defined for each od
for i=1:nb_od
    g_od_vector(i).od_logit_dominator=1e-5;% initialized logit dominator as a small nb
    for j=1:nb_path
        if g_od_vector(i).from_zone_id==g_path_vector(j).from_zone_id && g_od_vector(i).to_zone_id==g_path_vector(j).to_zone_id
         g_od_vector(i).od_logit_dominator...
             = g_od_vector(i).od_logit_dominator...
             +exp(-g_od_vector(i).estimated_theta*g_path_vector(j).path_travel_time-g_path_vector(j).path_cost);
        end
    end
end
%% == Step B: logit numerator function is defined for each path
for i=1:nb_od
    for j=1:nb_path
        if g_od_vector(i).from_zone_id==g_path_vector(j).from_zone_id && g_od_vector(i).to_zone_id==g_path_vector(j).to_zone_id
            g_path_vector(j).path_estimated_proportion...
                =exp(-g_od_vector(i).estimated_theta*g_path_vector(j).path_travel_time-g_path_vector(j).path_cost)...
                /g_od_vector(i).od_logit_dominator;
        end
    end
end


end

