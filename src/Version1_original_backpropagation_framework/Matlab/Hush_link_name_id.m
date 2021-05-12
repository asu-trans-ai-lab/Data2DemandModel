function [map_link_name_id] = Hush_link_name_id(g_link_vector)
[~,nb_link]=size(g_link_vector);
LINK_ID={};
LINK_NAME={};
for i=1:nb_link
    LINK_ID=[LINK_ID,g_link_vector(i).link_id];
    LINK_NAME=[LINK_NAME,num2str([g_link_vector(i).from_node_id,g_link_vector(i).to_node_id])];
end
map_link_name_id=containers.Map(LINK_NAME,LINK_ID);


end

