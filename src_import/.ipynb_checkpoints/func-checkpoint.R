loadMEUSE <- function(){
    
    data(meuse)
    coordinates(meuse) <- ~x+y
    proj4string(meuse) <- CRS("+proj=sterea +lat_0=52.15616055555555 +lon_0=5.38763888888889 +k=0.9999079 +x_0=155000 +y_0=463000 +ellps=bessel +towgs84=565.2369,50.0087,465.658,-0.406857,0.350733,-1.87035,4.0812 +units=m +no_defs")

    data(meuse.grid)
    coordinates(meuse.grid) = ~x+y
    proj4string(meuse.grid) <- CRS("+proj=sterea +lat_0=52.15616055555555 +lon_0=5.38763888888889 +k=0.9999079 +x_0=155000 +y_0=463000 +ellps=bessel +towgs84=565.2369,50.0087,465.658,-0.406857,0.350733,-1.87035,4.0812 +units=m +no_defs")
    gridded(meuse.grid) = TRUE
    
    meuse_list <- list(meuse,meuse.grid)
    names(meuse_list) <- c('meuse','meuse.grid')
    
    return(meuse_list)
}

pseudoR2 <- function(y_obs, y_pred){
    res <- y_obs-y_pred
    tmp <- 1-sum(res^2)/sum((y_obs-mean(y_obs))^2)
    return(tmp)
}

## Code from gstat
gstatkrg <- function(data, data.grid, formula, filename.out = NULL,filename.grid.out = NULL, var_model = "Sph"){

    # Compute the empirical variogram and fit model to the residuals
    fit <- lm(formula, data)
    vg <- variogram(formula, data)
    fit.vg <- fit.variogram(vg, vgm(var_model))
    data$res <- fit$res
    if(!is.empty(filename.out)) write.table(data, filename.out, sep = ",", col.names = TRUE, row.names = FALSE, quote = FALSE)

    # (Universal) Kriging
    krg <- krige(formula, data, data.grid, model = fit.vg)
    data.grid$mean <- krg$var1.pred
    data.grid$sd <- sqrt(krg$var1.var)
    if(!is.empty(filename.grid.out)) write.table(data.grid, filename.grid.out, sep = ",", col.names = TRUE, row.names = FALSE, quote = FALSE)
}

get_INLAspde_results <-  function(data, data.grid, mesh, model, stack,response_var, filename.out = NULL,filename.grid.out = NULL, sp_res = c(300,300)){ 

    ## Predictions
    data_out <- data.frame(data,model$summary.fitted.values[inla.stack.index(stack, tag = "train")$data, 1:5])
    data_out$pr2 <- pseudoR2(data_out[[response_var]],data_out$mean)
    data.grid_out <- data.frame(data.grid,model$summary.fitted.values[inla.stack.index(stack, tag = "pred")$data, 1:5])

    ## Spatial latent field
    proj = inla.mesh.projector(mesh, dims = sp_res)
    spatial_mean = inla.mesh.project(proj, model$summary.random[['spatial.field']][['mean']], dims = sp_res)
    spatial_sd = inla.mesh.project(proj, model$summary.random[['spatial.field']][['sd']], dims = sp_res)
    spatial_0.025quant = inla.mesh.project(proj, model$summary.random[['spatial.field']][['0.025quant']], dims = sp_res)
    spatial_0.5quant = inla.mesh.project(proj, model$summary.random[['spatial.field']][['0.5quant']], dims = sp_res)
    spatial_0.975quant = inla.mesh.project(proj, model$summary.random[['spatial.field']][['0.975quant']], dims = sp_res)
    spatial_mode = inla.mesh.project(proj, model$summary.random[['spatial.field']][['mode']], dims = sp_res)

    sp_out <- as.data.frame(cbind(x=proj$lattice$loc[,1],y=proj$lattice$loc[,2], spatial_mean = melt(spatial_mean)$value, spatial_SD = melt(spatial_sd)$value, 
                        spatial_0.025quant =melt(spatial_0.025quant)$value, 
                        spatial_0.5quant =melt(spatial_0.5quant)$value,
                        spatial_0.975quant =melt(spatial_0.975quant)$value, 
                        spatial_mode =melt(spatial_mode)$value))
    colnames(sp_out)[colnames(sp_out)=="spatial_SD"] <- "spatial_sd"

    if(!is.empty(filename.out)) write.table(data_out,filename.out, sep = ",", col.names = TRUE, row.names = FALSE, quote = FALSE)
    if(!is.empty(filename.grid.out)) write.table(data.grid_out,filename.grid.out, sep = ",", col.names = TRUE, row.names = FALSE, quote = FALSE)
    if(!is.empty(filename.out))  write.table(sp_out,gsub('.csv','_sp.csv',filename.out), sep = ",", col.names = TRUE, row.names = FALSE, quote = FALSE)

}

## Code from https://becarioprecario.bitbucket.io
INLAspde <- function(data, data.grid, family, response_var, predictors, data.crs,filename.out, filename.grid.out){

    #Define mesh
    bnd <- inla.nonconvex.hull(coordinates(data.grid),crs=data.crs)
    mesh <- inla.mesh.2d(loc = coordinates(data.grid), boundary = bnd,  cutoff = 100, max.edge = c(250, 500), offset = c(100, 250))

    png(gsub('.csv','_mesh.png',filename.out), width = 20, height = 20, units = 'cm', res = 300)
    plot(mesh, asp = 1, main = "")
    points(coordinates(data), pch = 21, bg = 'red', col = 'red', cex = 1)
    dev.off()

    #Create SPDE
    #sig0 = 0.1; rho0 = 0.1 #rho0 is typical range, sig0 typical sd 
    ##  (ρ_0, P(ρ < ρ_0)=p_ρ) where ρ is the spatial range of the random field.
    ##  (σ_0, P(σ > σ_0)=p_σ) where σ is the marginal standard deviation of the field)
    spde <- inla.spde2.pcmatern(mesh = mesh,  alpha = 2, constr=TRUE, prior.range=c(700,0.1), prior.sigma=c(0.2,0.1))
    s.index <- inla.spde.make.index(name = "spatial.field",n.spde = spde$n.spde)

    #Create data structure
    A.train <- inla.spde.make.A(mesh = mesh, loc = coordinates(data))
    stack.train <- inla.stack(data  = list(response_var = data[[response_var]]),
                                    A = list(A.train, 1),
                                    effects = list(c(s.index, list(Intercept = 1)),
                                                    data.frame(data) %>%                       
                                                    select(predictors) %>%                                                      
                                                    as.list()),
                                    tag = "train")

    #Create data structure for prediction
    A.pred <- inla.spde.make.A(mesh = mesh, loc = coordinates(data.grid))
    stack.pred <- inla.stack(data  = list(response_var = NA),
                                    A = list(A.pred, 1),
                                    effects = list(c(s.index, list(Intercept = 1)),
                                                    data.frame(data.grid) %>%                       
                                                    select(predictors) %>%                                                      
                                                    as.list()),
                                    tag = "pred")

    #Join stack
    stack.join <- inla.stack(stack.train, stack.pred)

    #Fit model
    ff <- as.formula(paste('response_var', paste(c('-1', 'Intercept', predictors, 'f(spatial.field, model = spde)'), collapse=" + "),sep='~'))
    model <- inla(ff, data = inla.stack.data(stack.join, spde = spde),
        family = family,
        control.predictor = list(A = inla.stack.A(stack.join), compute = TRUE,link = 1),
        control.compute = list(cpo = TRUE, dic = TRUE), verbose = TRUE)

    #Summary of results
    print(summary(model))

    get_INLAspde_results(data, data.grid, mesh, model, stack.join,response_var, filename.out = './data/meuse.INLAspde.csv', filename.grid.out = './data/meuse.grid.INLAspde.csv')

}
