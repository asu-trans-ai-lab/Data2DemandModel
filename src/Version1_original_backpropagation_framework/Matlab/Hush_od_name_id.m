function [map_od_name_id] = Hush_od_name_id(g_od_vector)
[~,nb_od]=size(g_od_vector);
OD_ID={};
OD_NAME={};
for i=1:nb_od
    OD_ID=[OD_ID,g_od_vector(i).od_id];
    OD_NAME=[OD_NAME,num2str([g_od_vector(i).from_zone_id,g_od_vector(i).to_zone_id])];
end
map_od_name_id=containers.Map(OD_NAME,OD_ID);
end

