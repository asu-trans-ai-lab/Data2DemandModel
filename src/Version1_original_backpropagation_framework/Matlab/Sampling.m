function [g_ozone_vector,g_od_vector,g_link_vector]=Sampling(g_ozone_vector,g_od_vector,g_link_vector,nb,sample_ozone,sample_od,sample_link)
[~,nb_ozone]=size(g_ozone_vector);
[~,nb_od]=size(g_od_vector);
[~,nb_link]=size(g_link_vector);

for o=1:nb_ozone
    g_ozone_vector(o).target_generation=sample_ozone{nb}(o).target_generation;
    g_ozone_vector(o).trip_rate=sample_ozone{nb}(o).trip_rate;
    g_ozone_vector(o).population=sample_ozone{nb}(o).population;
    g_ozone_vector(o).total_gamma=0;
end


for od=1:nb_od
    g_od_vector(od).target_demand=sample_od{nb}(od).target_demand;
    g_od_vector(od).od_target_gamma=sample_od{nb}(od).od_target_gamma;
end
for od=1:nb_od
    for o=1:nb_ozone
        if g_od_vector(od).from_zone_id==g_ozone_vector(o).ozone_id
            g_ozone_vector(o).total_gamma=g_ozone_vector(o).total_gamma+g_od_vector(od).od_target_gamma;
        end
    end
end
for od=1:nb_od
    for o=1:nb_ozone
        if g_od_vector(od).from_zone_id==g_ozone_vector(o).ozone_id
            g_od_vector(od).od_target_gamma=g_od_vector(od).od_target_gamma/g_ozone_vector(o).total_gamma;
        end
    end
end
for l=1:nb_link
    g_link_vector(l).target_flow=sample_link{nb}(l).target_flow;
end
end