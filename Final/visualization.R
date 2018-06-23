library(dplyr)
library(magrittr)
library(ggplot2)

####---資料載入整理---####
d=read.table("data.csv",sep=",",header=T,stringsAsFactor=F)

d %<>% mutate(home_type=ifelse(grepl("@",matchup),2,1),
              remain_sec=minutes_remaining*60+seconds_remaining,
              game_date=as.Date(game_date)) %>% 
  select(-team_id,-team_name,-loc_x,-loc_y,-minutes_remaining,-seconds_remaining,-matchup)

trd=d %>% filter(is.na(shot_made_flag)==FALSE)
ted=d %>% filter(is.na(shot_made_flag)==TRUE)

####---ggplot主題設定---####
themep=theme(
  plot.title=element_text(face="bold",size=18),
  legend.title=element_text(size=16),
  legend.text=element_text(size=14),
  axis.title.x=element_text(size=14),
  axis.title.y=element_text(size=14),
  axis.text.x=element_text(size=12),
  axis.text.y=element_text(size=12)
  
)

####---圖1射籃型態圓餅圖和長條圖---####
p1=trd %>% group_by(combined_shot_type) %>% summarise(p=round(n()/nrow(trd),3)*100) %>% arrange(p) %>% ungroup %>% mutate(pos=cumsum(p)-0.5*p) %>%
  ggplot(.,aes(x="",y=p,fill=combined_shot_type))+
  geom_bar(width=1,stat="identity")+
  coord_polar(theta="y")+
  geom_text(aes(y=pos,label=c("","","",paste0(p[4:6],"%"))),size=6,fontface="bold")+
  scale_fill_discrete(name="combined_shot_type",labels=c("Bank Shot ( 0.5% )","Dunk","Hook Shot ( 0.5% )","Jump Shot","Layup","Tip Shot ( 0.6% )"))+
  ggtitle("射籃型態比例")+
  theme_minimal()+
  theme(
    plot.title=element_text(face="bold",size=18),
    axis.title.x=element_blank(),
    axis.title.y=element_blank(),
    panel.grid=element_blank(),
    axis.text=element_blank(),
    legend.title=element_text(size=16),
    legend.text=element_text(size=14)
  )

p2=trd %>% group_by(combined_shot_type,shot_made_flag) %>% tally %>% group_by(combined_shot_type) %>% mutate(p=round(n/sum(n),3)*100) %>% 
  ggplot(.,aes(x=combined_shot_type,y=p,fill=factor(shot_made_flag)))+
  geom_bar(stat="identity",position="dodge")+
  geom_text(aes(label=paste0(p,"%")),position=position_dodge(width=0.9),size=4,colour="white",vjust=1,fontface="bold")+
  scale_fill_discrete(name="shot_made_flag",breaks=c("0","1"),labels=c("Failed","Made"))+
  ggtitle("各投籃型態命中率(%)")+
  ylab("Proportion")+
  themep

gridExtra::grid.arrange(p1,p2,nrow=1)


####---圖2射籃位置分布與命中率---####
new_area_p=trd %>% mutate(new_area=paste0(shot_zone_area," & ",shot_zone_basic)) %>% group_by(new_area) %>% summarise(total=n(),made=sum(shot_made_flag),p=round(sum(shot_made_flag)/n(),3))
new_area_loc=trd %>% mutate(new_area=paste0(shot_zone_area," & ",shot_zone_basic)) %>% group_by(new_area) %>% summarise(medlat=median(lat),medlon=median(lon))
new_area_merge=left_join(new_area_p,new_area_loc,"new_area"="new_area") %>% mutate(madelabel=paste0(p*100,"%","\n"," (",made,"/",total,")"))

trd %>% mutate(newarea=paste0(shot_zone_area," & ",shot_zone_basic)) %>% 
  ggplot(aes(y=lat,x=lon,colour=factor(newarea)))+
  geom_point()+
  annotate("text",y=new_area_merge$medlat,x=new_area_merge$medlon,label=new_area_merge$madelabel,size=5,fontface="bold")+
  scale_colour_discrete(name="shot_zone_area & shot_zone_basic")+
  ylab("Latitude")+
  xlab("Longitude")+
  ggtitle("投籃位置與命中率")+
  themep


####---圖3射籃距離分布與命中率---####
new_dis_p=trd %>% mutate(new_dis=paste0(shot_zone_range)) %>% group_by(new_dis) %>% summarise(total=n(),made=sum(shot_made_flag),p=round(sum(shot_made_flag)/n(),3))
new_dis_loc=trd %>% mutate(new_dis=paste0(shot_zone_range)) %>% group_by(new_dis) %>% summarise(medlat=median(lat),medlon=median(lon))
new_dis_merge=left_join(new_dis_p,new_dis_loc,"new_dis"="new_dis") %>% mutate(madelabel=paste0(p*100,"%","\n"," (",made,"/",total,")"))

trd %>% mutate(PointColour=ifelse(shot_distance %in% c(8,16,24),shot_distance,1))%>% 
  ggplot(aes(y=lat,x=lon,colour=factor(PointColour)))+
  geom_point()+
  annotate("text",y=new_dis_merge$medlat,x=new_dis_merge$medlon,label=new_dis_merge$madelabel,size=5,fontface="bold")+
  scale_colour_discrete(name="shot distance",labels=c("all shots","8 ft.","16 ft.","24 ft."))+
  ylab("Latitude")+
  xlab("Longitude")+
  ggtitle("投籃位置與命中率")+
  themep

####---圖4每季得分與2pts.3pts.命中率---####
trd %<>% mutate(shot_type_pts=ifelse(grepl("2",shot_type),2,3))
trd %<>% mutate(pts=as.numeric(shot_type_pts*shot_made_flag)) %>% 
  select(-shot_type_pts)

points = trd %>% arrange(season) %>% group_by(season) %>% mutate(pts=sum(pts))
points %<>% distinct(season, pts)

p3=points %>%
  ggplot(aes(x=season,y=pts))+
  geom_bar(stat="identity")+
  geom_text(aes(x=season,y=pts,label=pts),vjust=-1,color="red",fontface="bold")+
  ggtitle("Kobe每季總得分")+
  ylab("Points")+
  themep +
  theme(
    strip.text.x=element_text(size=14,face="bold"),
    axis.text.x=element_text(angle=45,vjust=0.3) 
  )


p4=trd %>% group_by(season,shot_type,shot_made_flag) %>% tally %>% group_by(season,shot_type) %>% mutate(p=ifelse(shot_made_flag==1,paste0(round(n/sum(n),3)*100,"%"),"")) %>%
  ungroup %>% arrange(season,shot_type,desc(shot_made_flag)) %>% group_by(season,shot_type) %>% mutate(loc=cumsum(n)-0.5*n,sumn=sum(n)) %>% ungroup %>%
  ggplot(aes(x=season,y=n,fill=factor(shot_made_flag)))+
  geom_bar(stat="identity",position="stack")+
  geom_text(aes(x=season,y=loc,label=p),fontface="bold",size=4,color="red")+
  geom_text(aes(x=season,y=sumn,label=sumn),vjust=-1,fontface="bold",color="blue")+
  scale_fill_manual(name="made or not",breaks=c("0","1"),labels=c("Failed","Made"),values=RColorBrewer::brewer.pal(3,"Accent")[2:1])+
  facet_grid(~shot_type)+
  ggtitle("Kobe每季2pts和3pts命中率")+
  ylab("投射次數")+
  themep+
  theme(
    strip.text.x=element_text(size=14,face="bold"),
    axis.text.x=element_text(angle=45,vjust=0.3)
  )

gridExtra::grid.arrange(p3,p4,nrow=2)
