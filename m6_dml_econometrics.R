#### ---------------------------------------------------------------------------
# Machine Learning in Econometrics: Dataproject
# by Alexander Vogt
set.seed(1234)

rm(list=ls())
## Set working directory
setwd()
options(scipen=999)

library(tidyverse)
library(haven)
library(glmnet)
library(hdm)
library(ggplot2)
library(plm)
library(grf)
library(randomForest)
library(caret)
library(lmtest)
library(sandwich)
library(stargazer)
library(mlr3)
library(mlr3learners)
library(DoubleML)
library(ranger)



################################################################################
########################## Useful Functions ####################################
################################################################################

### Double Lasso Function by Partialling-Out

DML_Lasso <- function(Y, D, W) {
  
  # Create Empty Data Frame
  
  z <- matrix(nrow = 2, ncol= 1)
  
  # Partial Out W from Y
  
  fit.lasso.Y <- cv.glmnet(W, Y)
  fitted.lasso.Y <- predict(fit.lasso.Y, newx = W, s="lambda.min")
  Y.tilde <- Y - fitted.lasso.Y
  
  # Partial Out W from D
  
  fit.lasso.D <- cv.glmnet(W, D)
  fitted.lasso.D <- predict(fit.lasso.D, newx = W, s="lambda.min")           
  D.tilde <- D - fitted.lasso.D  
  
  # Regress log wage on Female with OLS
  
  final <- lm(Y.tilde ~ -1 + D.tilde)
  
  # Extract Values
  
  #final_out <- summary(final)
  #z[1,1] <- coef(final)[2]
  #z[2,1] <- final_out$coefficients[2,2]
  
  return(final)
}

### Function to plot sparsity
plot_sparsity <- function(fitted_values) {
  # Extract coefficients and feature names
  coefficients <- coef(fitted_values)[-1]
  feature_names <-  1:length(coefficients)
  
  # Create a data frame to display coefficients
  coefficients_df <- data.frame(Feature = feature_names, Coefficient = coefficients)
  
  # Sort the data frame by coefficient magnitudes
  coefficients_df <- coefficients_df[order(abs(coefficients_df$Coefficient), decreasing = TRUE), ]
  coefficients_df$abscoef <- abs(coefficients_df$Coefficient) 
  # Visualize coefficients with a bar plot
  ggplot(coefficients_df, aes(x = factor(Feature, levels = coefficients_df$Feature[order(-coefficients_df$abscoef)]), y = abscoef)) +
    geom_point(stat = "identity", fill = "steelblue") +
    labs(x = "Coefficient", y = "Magnitude", title = "Sparsity: Lasso Coefficients") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    scale_x_discrete(breaks = unique(coefficients_df$Feature)[seq(1, length(unique(coefficients_df$Feature)), 25)]) +
    coord_cartesian(xlim = c(0, 50)) +
    theme_bw()
}

################################################################################
########################## Data Cleaning #######################################
################################################################################

### ------  Load Dataset  

gender <- read_csv("genderinequality.csv")

### ------ Calculate Log Wage, Deal with NA's and new variables

# Data is prepared for a Difference-in-Difference approach with a Double Machine
# Learning Approach using Double Lasso, Post-Double Selection Lasso
# Therefore, a modification of the data is necessary in order to split into 
# treatment and control group. While the the Post Variable indicates before and 
# after the introduction of the policy

# Create the Treatment and Control Group

  # Check if Data is balanced:
  # Result: True, every individual is observed twice. Once in 2005 and 2010.

      is.pbalanced(gender) 
  
  # Create Treament Group Variable in order to apply the DiD Approach
      
    # Create Treatment Group (TG) Indicator based on the year 2010
      treat_10 <- gender %>% 
              filter(year == 2010) %>%
              arrange(id) %>% 
              mutate(TG = treat)
      
      z_10 <- treat_10 %>%  select(id, TG)
      
    # As all individuals are observed in both years adding the TG from 2010
    # to the data in 2005 is a valid process to create the TG variable.
    # Now, ID and Treat are matched in both years.
      
      treat_05 <- gender %>% 
              filter(year == 2005) %>%
              arrange(id) %>% 
              mutate(TG = treat_10$TG)
      z_05 <- treat_05 %>%  select(id, TG)

      gender <- bind_rows(treat_05, treat_10)
      
    # If ID and Treat are correct matched, a sub-data frame of ID and Treat
    # must be identical for both year.
    # Result: True
      
      all.equal(z_05, z_10)
      
      
gender <- gender %>%
  mutate(hr_wage = if_else(wage > 0, wage/hours, 0)) %>% 
  mutate(log_hr_wage = if_else(hr_wage > 0, log(wage/hours), 0)) %>%
  filter(log_hr_wage != Inf) %>% 
  mutate(meduc = replace_na(meduc, floor(mean(meduc,na.rm=T))),
         feduc = replace_na(feduc, floor(mean(feduc, na.rm=T))),
         brthord = replace_na(brthord, floor(mean(feduc, na.rm=T))),
         post = if_else(year == 2010, 1, 0),
         educ2 = educ^2,
         educ3 = educ^3,
         tenure2 = tenure^2,
         tenure3 = tenure^3,
         exper2 = exper^2,
         exper3 = exper^3,
         age2 = age^2,
         age3 = age^3,
         IQ2 = IQ^2,
         IQ3 = IQ^3
         ) %>% 
  mutate(post_treat_fe = post * TG * female,
         post_treat = post * TG,
         post_fe = post * female,
         treat_fe = treat * female,
         TG_fe = TG * female)

### ------- Gender Pay Gap Dataset: GPG

### Determine Outliers in Log Hourly Wage

  ## Quantile and IQR

q25 <- quantile(gender$log_hr_wage, probs = 0.25, na.rm = FALSE)
q75 <- quantile(gender$log_hr_wage, probs = 0.75, na.rm = FALSE)
iqr <- IQR(gender$log_hr_wage)

  ## Calculate Upper and Lower by adding 1.5 *  interquartile range Range 
  ##on top/below of the 75%-quartile/25%-quartile

up <- q75 + 1.5 * iqr
low <- q25 - 1.5 * iqr 
  
  ## Drop Outliers, all unemployed and Employment-Variable

GPG <- gender %>%  
  filter(emp == 1) %>% 
  filter(log_hr_wage < up & log_hr_wage > low) %>% 
  select(-emp)

### ------ GPG Sub-Dataset for 2005 and 2010
  
  GPG05 <- GPG %>%  filter(year == 2005)
  
  GPG10 <- GPG %>%  filter(year == 2010)  

### ------- Gender Employment Gap Dataset: GEmp

  GEGap <- gender 

  GEGap05 <- gender %>%  filter(year == 2005) 

  GEGap10 <- gender %>%  filter(year == 2010)

################################################################################
############################ Descriptive Statistik ############################
################################################################################

### Correlation Table with only half of the matrix being displayed -------------
    
  
summary_statistics <-  GPG %>% 
    filter(year == 2010) %>% 
    select(IQ, KWW, educ, exper, tenure, age, sibs, brthord, meduc, feduc, log_hr_wage) %>% 
    as.data.frame()

colnames(summary_statistics) <- c("IQ", "KWW", "Education", "Experience", "Tenure",
                                  "Age", "Siblings", "Birthorder", "Mother Educ.",
                                  "Father Educ.", "Log Hourly Wage")
  
stargazer(summary_statistics, out="summary_statistics.html",
          title = "Tbl. 1: Summary Statistics")

shares <- gender %>%
          filter(year == 2010) %>% 
          select(female, urban, black, married, south) %>% 
          summarise(Female = mean(female) * 100, Urban = mean(urban) *100, 
                    Black = mean(black) *100 ,South = mean(south) * 100,
                    Married = mean(married)*100) %>% 
          as.data.frame()

stargazer(shares,
          summary.stat = c("Mean"),
          column.labels = c("Statistic", "Shares"),
          title = "Tbl. 2: Shares",
          out="Percentage Share of Parts of the Population.html")

      ### Wage: Boxplot and Histogram
boxplot_before <- gender %>% 
  filter(emp==1) %>% 
  filter(log_hr_wage >= up | log_hr_wage <= low) %>% 
  ggplot(aes(log_hr_wage)) +
  geom_boxplot()

boxplot_after <- gender %>% 
  ggplot() +
  aes(y=log_hr_wage) +
  geom_boxplot()

GPG %>% 
  group_by(year) %>% 
  summarise(year_mean = mean(log_hr_wage))

hist_wage <- GPG %>% 
  mutate(year_fac = as.factor(year)) %>% 
  #filter(year == 2005) %>% 
  ggplot(aes(x=log_hr_wage, color = year_fac)) +
  guides(fill = guide_legend(title = "Year")) +
  geom_density(alpha=0.4) +
  ggtitle("Wage Distribution by Year") +
  ylab("Density") +
  xlab("Log Hourly Wage") +
  labs(caption = "(Graph 1)") +
  theme_bw()

ggsave(hist_wage, filename = "hist_wage.png")


################################################################################
############################# Formula ##########################################
################################################################################

### Simple Control Set ---------------------------------------------------------

form_simple <- as.formula(
                    ~ -1  + post_treat + post_fe + KWW + age + sibs + brthord 
                    + sibs +  married + black + south + urban + meduc
                    + feduc + educ + educ2 + educ3 + tenure + tenure2 
                    + age + age2  + exper + exper2
                    )

### Fexible Control Set --------------------------------------------------------

form_flex <- as.formula(
  ~ -1  + post_treat + post_fe + KWW + age + sibs + brthord 
  + sibs +  married + black + south + urban + meduc
  + feduc + educ + educ2 + educ3 + tenure + tenure2 
  + age + age2  + exper + exper2 + 
    (KWW + age + sibs + brthord + sibs +  married + black + south 
     + urban + meduc + feduc + educ + educ2 + educ3 + tenure 
     + tenure2 + age + age2  + exper + exper2)^2
)

################################################################################
############################# Gender Pay Gap ###################################
################################################################################

######
### Double Lasso ===============================================================
######

#### Year 2005 -----------------------------------------------------------------

  # Simple ----------
W05s <- model.matrix(form_simple , data = GPG05) 

D_GPG_05S <- GPG05$TG_fe
Y_GPG_05S <- GPG05$log_hr_wage

GPG_result_05S <- coeftest(DML_Lasso(Y_GPG_05S, D_GPG_05S, W05s))
GPG_result_05S

  # Flexible ---------
W05 <- model.matrix(form_flex , data = GPG05) 

D_GPG_05 <- GPG05$TG_fe
Y_GPG_05 <- GPG05$log_hr_wage

GPG_result_05 <- coeftest(DML_Lasso(Y_GPG_05, D_GPG_05, W05))
GPG_result_05

#### Year 2010 -----------------------------------------------------------------

  # Simple ----------
W10S <- model.matrix(form_simple,data = GPG10)

D_GPG_10S <- GPG10$TG_fe
Y_GPG_10S <- GPG10$log_hr_wage

GPG_result_10S <- coeftest(DML_Lasso(Y_GPG_10S, D_GPG_10S, W10S))
GPG_result_10S

  #Flexible ---------

W10 <- model.matrix(form_flex
                    ,data = GPG10)

D_GPG_10 <- GPG10$TG_fe
Y_GPG_10 <- GPG10$log_hr_wage

GPG_result_10 <- coeftest(DML_Lasso(Y_GPG_10, D_GPG_10, W10))
GPG_result_10

#### DiD-Approach for ATE ------------------------------------------------------

  ### Simple ---------
X1AllS <- model.matrix(form_simple, data = GPG)
D_AllS <- GPG$post_treat_fe
Y_AllS <- GPG$log_hr_wage

GPG_result_DiDS <- coeftest(DML_Lasso(Y_AllS, D_AllS, X1AllS))


  ### Flexible -------
X1All <- model.matrix(form_flex, data = GPG)
D_All <- GPG$post_treat_fe
Y_All <- GPG$log_hr_wage

GPG_result_DiD <- coeftest(DML_Lasso(Y_All, D_All, X1All))

#### Tables for all previous 
  
  ### Table: 2005 ---------
table_05_GPG <- stargazer(GPG_result_05S, GPG_result_05,
                          dep.var.caption = "GPG 2005",
                          dep.var.labels.include = TRUE,
                          dep.var.labels = "Log Hourly Wage",
                          column.labels = c("Simple", "Flexible"),
                          covariate.labels = "GPG",
                          type = "html",
                          out="GPG_05.html"
)
  ### Table: 2010 ---------
table_10_GPG <- stargazer(GPG_result_10S, GPG_result_10,
                          dep.var.caption = "GPG 2010",
                          dep.var.labels.include = TRUE,
                          dep.var.labels = "Log Hourly Wage",
                          column.labels = c("Simple", "Flexible"),
                          covariate.labels = "GPG",
                          type = "html",
                          out="GPG_10.html"
)

  ### Table: Panel Data ---

table_All_GPG <- stargazer(GPG_result_DiDS, GPG_result_DiD,
                          dep.var.caption = "Panel Data",
                          dep.var.labels.include = TRUE,
                          dep.var.labels = "Log Hourly Wage",
                          column.labels = c("Simple", "Flexible"),
                          covariate.labels = "Treatment Effect",
                          type = "html",
                          out="GPG_PD.html"
)

### Calculation via every single step for Sparsity-Plots =======================

# Partial Out W from Y


fit.lasso.Y.all <- cv.glmnet(X1All, Y_All)
fitted.lasso.Y.all <- predict(fit.lasso.Y.all, newx = X1All, s="lambda.min")
Y.tilde.all <- Y_All - fitted.lasso.Y.all

# Partial Out W from D

fit.lasso.D.all <- cv.glmnet(X1All, D_All)
fitted.lasso.D.all <- predict(fit.lasso.D.all, newx = X1All, s="lambda.min")           
D.tilde.all <- D_All - fitted.lasso.D.all 

summary(lm(Y.tilde.all ~  D.tilde.all))

model <- lm(Y.tilde.all ~ D.tilde.all)

coeftest(model, vcov = vcovHC(model, type = "HC2"))


#### Sparsity Plots ------------------------------------------------------------

sparsity_wage <- plot_sparsity(fit.lasso.Y.all)
ggsave(sparsity_wage, filename = "sparsity_wage.png")

sparsity_D <- plot_sparsity(fit.lasso.D.all)
ggsave(sparsity_D, filename = "sparsity_D.png")


#### Heterogeneity Analyse -----------------------------------------------------

  #### Black----------------

GPG_B <- GPG %>%  filter(black == 1)
X_GPG_B <- model.matrix(form_flex, data = GPG_B)
Y_GPG_B <- GPG_B$log_hr_wage
D_GPG_B <- GPG_B$post_treat_fe

GPG_result_DiD_B <- coeftest(DML_Lasso(Y_GPG_B, D_GPG_B, X_GPG_B))

  #### White ----------------

GPG_W <- GPG %>%  filter(black == 0)
X_GPG_W <- model.matrix(form_flex, data = GPG_W)
Y_GPG_W <- GPG_W$log_hr_wage
D_GPG_W <- GPG_W$post_treat_fe

GPG_result_DiD_W <- coeftest(DML_Lasso(Y_GPG_W, D_GPG_W, X_GPG_W))

  #### Urban ----------------

GPG_U <- GPG %>%  filter(urban == 1)
X_GPG_U <- model.matrix(form_flex, data = GPG_U)
Y_GPG_U <- GPG_U$log_hr_wage
D_GPG_U <- GPG_U$post_treat_fe

GPG_result_DiD_U <- coeftest(DML_Lasso(Y_GPG_U, D_GPG_U, X_GPG_U))

  ### Rural Area ------------

GPG_RA <- GPG %>%  filter(urban == 0)
X_GPG_RA <- model.matrix(form_flex, data = GPG_RA)
Y_GPG_RA <- GPG_RA$log_hr_wage
D_GPG_RA <- GPG_RA$post_treat_fe

GPG_result_DiD_RA <- coeftest(DML_Lasso(Y_GPG_RA, D_GPG_RA, X_GPG_RA))

  ### South -----------------

GPG_S <- GPG %>%  filter(south == 1)
X_GPG_S <- model.matrix(form_flex, data = GPG_S)
Y_GPG_S <- GPG_S$log_hr_wage
D_GPG_S<- GPG_S$post_treat_fe

GPG_result_DiD_S <- coeftest(DML_Lasso(Y_GPG_S, D_GPG_S, X_GPG_S))

  ### North-------------------

GPG_N <- GPG %>%  filter(south == 0)
X_GPG_N <- model.matrix(form_flex, data = GPG_N)
Y_GPG_N <- GPG_N$log_hr_wage
D_GPG_N <- GPG_N$post_treat_fe

GPG_result_DiD_N <- coeftest(DML_Lasso(Y_GPG_N, D_GPG_N, X_GPG_N))

  ### Married -----------------

GPG_M <- GPG %>%  filter(married == 1)
X_GPG_M <- model.matrix(form_flex, data = GPG_M)
Y_GPG_M <- GPG_M$log_hr_wage
D_GPG_M <- GPG_M$post_treat_fe

GPG_result_DiD_M <- coeftest(DML_Lasso(Y_GPG_M, D_GPG_M, X_GPG_M))

  ### Not Married -------------

GPG_NM <- GPG %>%  filter(married == 0)
X_GPG_NM <- model.matrix(form_flex, data = GPG_NM)
Y_GPG_NM <- GPG_NM$log_hr_wage
D_GPG_NM <- GPG_NM$post_treat_fe

GPG_result_DiD_NM <- coeftest(DML_Lasso(Y_GPG_NM, D_GPG_NM, X_GPG_NM))

  ### Heterogeneity Table

table_BW_GPG <- stargazer(GPG_result_DiD_B, GPG_result_DiD_W,
                      dep.var.caption = "Heterogeneity (1)",
                      dep.var.labels.include = TRUE,
                      dep.var.labels = "Log Hourly Wage",
                      column.labels = c("Black", "White"),
                      covariate.labels = "Treatment Effect",
                      type = "html",
                      out="GPG_BW.html")

table_URA_GPG <- stargazer(GPG_result_DiD_U, GPG_result_DiD_RA,
                       dep.var.caption = "Heterogeneity (2)",
                       dep.var.labels.include = TRUE,
                       dep.var.labels = "Log Hourly Wage",
                       column.labels = c("Urban", "Rural"),
                       covariate.labels = "Treatment Effect",
                       type = "html",
                       out="GPG_URA.html"
                        )

table_SN_GPG <- stargazer(GPG_result_DiD_S, GPG_result_DiD_N,
                       dep.var.caption = "Heterogeneity (3)",
                       dep.var.labels.include = TRUE,
                       dep.var.labels = "Log Hourly Wage",
                       column.labels = c("South", "North"),
                       covariate.labels = "Treatment Effect",
                       type = "html",
                       out="GPG_SN.html"
)


table_MNM_GPG <- stargazer(GPG_result_DiD_M, GPG_result_DiD_NM,
                           dep.var.caption = "Heterogeneity (4)",
                           dep.var.labels.include = TRUE,
                           dep.var.labels = "Log Hourly Wage",
                           column.labels = c("Married", "Not M."),
                           covariate.labels = "Treatment Effect",
                           type = "html",
                           out="GPG_MNM.html"
                           )


######
### Double Machine Learning with RandomForest ==================================
######

  ### Prepare Data for DoubleML-Package

    X1All
    X1All_df <- X1All %>%  as.data.frame()
    X1All_dt <- X1All_df %>% 
      mutate(log_hr_wage = GPG$log_hr_wage,
             post_treat_fe = GPG$post_treat_fe) %>%
      rename_with(~ tolower(gsub(":", "_", .x, fixed = TRUE))) %>% 
      as.data.table()
      var_base_test <- head(names(X1All_dt), -2)

  ### Save Data for the Analyse
    data_dml_GPG <- DoubleMLData$new(X1All_dt,
                                    y_col = "log_hr_wage",
                                    d_cols = "post_treat_fe",
                                    x_cols = var_base_test)
  


  ### Actual Procedure of the DML with Random Forest

    randomForest_GPG <- lrn("regr.ranger",
                             max.depth = 5,
                             mtry = 5,
                             min.node.size = 7)

    randomForest_class_GPG <- lrn("classif.ranger", 
                                   max.depth = 5,
                                   mtry = 5,
                                    min.node.size = 7)


    dml_plr_forest_GPG <- DoubleMLPLR$new(data_dml_GPG,
                                 ml_l = randomForest_GPG,
                                 ml_m = randomForest_class_GPG,
                                 n_folds = 10)

    dml_plr_forest_GPG$fit()
    dml_plr_forest_GPG$summary()

  ### Save SE and Coefficient in data frame 

    RF_GPG_Coef <- dml_plr_forest_GPG$coef
    RF_GPG_SE <- dml_plr_forest_GPG$se 
    
    table_coef_GPG <- rbind(RF_GPG_Coef, RF_GPG_SE)
    rownames(table_coef_GPG) <- c("Coefficient", "SE")
    colnames(table_coef_GPG) <- c("ATE on GPG")
    table_coef_GPG

################################################################################
###################### Gender Employment Gap ###################################
################################################################################

######
### Double Lasso (GEG) =========================================================
######

#### Year 2005 -----------------------------------------------------------------

# Simple ----------
W05s_Emp <- model.matrix(form_simple , data = GEGap) 

D_GEGap_05S <- GEGap$TG_fe
Y_GEGap_05S <- GEGap$log_hr_wage

GEGap_result_05S <- coeftest(DML_Lasso(Y_GEGap_05S, D_GEGap_05S, W05s_Emp))
GEGap_result_05S

# Flexible ---------
W05_Emp <- model.matrix(form_flex , data = GEGap05) 

D_GEGap_05 <- GEGap05$TG_fe
Y_GEGap_05 <- GEGap05$log_hr_wage

GEGap_result_05 <- coeftest(DML_Lasso(Y_GEGap_05, D_GEGap_05, W05_Emp))
GEGap_result_05

#### Year 2010 -----------------------------------------------------------------

# Simple ----------
W10S_Emp <- model.matrix(form_simple,data = GEGap10)

D_GEGap_10S <- GEGap10$TG_fe
Y_GEGap_10S <- GEGap10$log_hr_wage

GEGap_result_10S <- coeftest(DML_Lasso(Y_GEGap_10S, D_GEGap_10S, W10S_Emp))
GEGap_result_10S

#Flexible ---------

W10_Emp <- model.matrix(form_flex
                    ,data = GEGap10)

D_GEGap_10 <- GEGap10$TG_fe
Y_GEGap_10 <- GEGap10$log_hr_wage

GEGap_result_10 <- coeftest(DML_Lasso(Y_GEGap_10, D_GEGap_10, W10_Emp))
GEGap_result_10

#### DiD-Approach for ATE ------------------------------------------------------

### Simple ---------
X1AllS_E <- model.matrix(form_simple, data = GEGap)
D_AllS_E <- GEGap$post_treat_fe
Y_AllS_E <- GEGap$log_hr_wage

GEGap_result_DiDS <- coeftest(DML_Lasso(Y_AllS_E, D_AllS_E, X1AllS_E))


### Flexible -------
X1All_E <- model.matrix(form_simple, data = GEGap)
D_All_E <- GEGap$post_treat_fe
Y_All_E <- GEGap$log_hr_wage

GEGap_result_DiD <- coeftest(DML_Lasso(Y_All_E, D_All_E, X1All_E))

#### Tables for all previous 

### Table: 2005 ---------
table_05_GEGap <- stargazer(GEGap_result_05S, GEGap_result_05,
                          dep.var.caption = "GEGap 2005",
                          dep.var.labels.include = TRUE,
                          dep.var.labels = "Employment",
                          column.labels = c("Simple", "Flexible"),
                          covariate.labels = "GEG",
                          type = "html",
                          out="GEGap_05.html"
)
### Table: 2010 ---------
table_10_GEGap <- stargazer(GEGap_result_10S, GEGap_result_10,
                          dep.var.caption = "GEGap 2010",
                          dep.var.labels.include = TRUE,
                          dep.var.labels = "Employment",
                          column.labels = c("Simple", "Flexible"),
                          covariate.labels = "GEG",
                          type = "html",
                          out="GEGap_10.html"
)

### Table: Panel Data ---

table_All_GEGap <- stargazer(GEGap_result_DiDS, GEGap_result_DiD,
                           dep.var.caption = "Panel Data",
                           dep.var.labels.include = TRUE,
                           dep.var.labels = "Employment",
                           column.labels = c("Simple", "Flexible"),
                           covariate.labels = "Treatment Effect",
                           type = "html",
                           out="GEGap_PD.html"
)

### Calculation via every single step for Sparsity-Plots =======================

# Partial Out W from Y

set.seed(1234)
fit.lasso.Y.all_E <- cv.glmnet(X1All_E, Y_All_E)
fitted.lasso.Y.all_E <- predict(fit.lasso.Y.all_E, newx = X1All_E, s="lambda.min")
Y.tilde.all_E <- Y_All - fitted.lasso.Y.all_E

# Partial Out W from D

fit.lasso.D.all_E <- cv.glmnet(X1All_E, D_All_E)
fitted.lasso.D.all_E <- predict(fit.lasso.D.all_E, newx = X1All_E, s="lambda.min")           
D.tilde.all_E <- D_All_E - fitted.lasso.D.all_E 

model <- lm(Y.tilde.all_E ~ D.tilde.all_E)

coeftest(model, vcov = vcovHC(model, type = "HC2"))


#### Sparsity Plots ------------------------------------------------------------

plot_sparsity(fit.lasso.Y.all_E)

plot_sparsity(fit.lasso.D.all_E)


#### Heterogeneity Analyse -----------------------------------------------------

  #### Black----------------

GEGap_B <- GEGap %>%  filter(black == 1)
X_GEGap_B <- model.matrix(form_flex, data = GEGap_B)
Y_GEGap_B <- GEGap_B$log_hr_wage
D_GEGap_B <- GEGap_B$post_treat_fe

GEGap_result_DiD_B <- coeftest(DML_Lasso(Y_GEGap_B, D_GEGap_B, X_GEGap_B))

  #### White ----------------

GEGap_W <- GEGap %>%  filter(black == 0)
X_GEGap_W <- model.matrix(form_flex, data = GEGap_W)
Y_GEGap_W <- GEGap_W$log_hr_wage
D_GEGap_W <- GEGap_W$post_treat_fe

GEGap_result_DiD_W <- coeftest(DML_Lasso(Y_GEGap_W, D_GEGap_W, X_GEGap_W))

  #### Urban ----------------

GEGap_U <- GEGap %>%  filter(urban == 1)
X_GEGap_U <- model.matrix(form_flex, data = GEGap_U)
Y_GEGap_U <- GEGap_U$log_hr_wage
D_GEGap_U <- GEGap_U$post_treat_fe

GEGap_result_DiD_U <- coeftest(DML_Lasso(Y_GEGap_U, D_GEGap_U, X_GEGap_U))

  ### Rural Area ------------

GEGap_RA <- GEGap %>%  filter(urban == 0)
X_GEGap_RA <- model.matrix(form_flex, data = GEGap_RA)
Y_GEGap_RA <- GEGap_RA$log_hr_wage
D_GEGap_RA <- GEGap_RA$post_treat_fe

GEGap_result_DiD_RA <- coeftest(DML_Lasso(Y_GEGap_RA, D_GEGap_RA, X_GEGap_RA))

  ### South -----------------

GEGap_S <- GEGap %>%  filter(south == 1)
X_GEGap_S <- model.matrix(form_flex, data = GEGap_S)
Y_GEGap_S <- GEGap_S$log_hr_wage
D_GEGap_S<- GEGap_S$post_treat_fe

GEGap_result_DiD_S <- coeftest(DML_Lasso(Y_GEGap_S, D_GEGap_S, X_GEGap_S))

  ### North ------------------

GEGap_N <- GEGap %>%  filter(south == 0)
X_GEGap_N <- model.matrix(form_flex, data = GEGap_N)
Y_GEGap_N <- GEGap_N$log_hr_wage
D_GEGap_N <- GEGap_N$post_treat_fe

GEGap_result_DiD_N <- coeftest(DML_Lasso(Y_GEGap_N, D_GEGap_N, X_GEGap_N))

  ### Married -----------------

GEGap_M <- GEGap %>%  filter(married == 1)
X_GEGap_M <- model.matrix(form_flex, data = GEGap_M)
Y_GEGap_M <- GEGap_M$log_hr_wage
D_GEGap_M <- GEGap_M$post_treat_fe

GEGap_result_DiD_M <- coeftest(DML_Lasso(Y_GEGap_M, D_GEGap_M, X_GEGap_M))

  ### Not Married -------------

GEGap_NM <- GEGap %>%  filter(married == 0)
X_GEGap_NM <- model.matrix(form_flex, data = GEGap_NM)
Y_GEGap_NM <- GEGap_NM$log_hr_wage
D_GEGap_NM <- GEGap_NM$post_treat_fe

GEGap_result_DiD_NM <- coeftest(DML_Lasso(Y_GEGap_NM, D_GEGap_NM, X_GEGap_NM))

### Heterogeneity Table --------------------------------------------------------

table_BW_GEGap <- stargazer(GEGap_result_DiD_B, GEGap_result_DiD_W,
                          dep.var.caption = "Heterogeneity (1)",
                          dep.var.labels.include = TRUE,
                          dep.var.labels = "Employment",
                          column.labels = c("Black", "White"),
                          covariate.labels = "Treatment Effect",
                          type = "html",
                          out="GEGap_BW.html")

table_URA_GEGap <- stargazer(GEGap_result_DiD_U, GEGap_result_DiD_RA,
                           dep.var.caption = "Heterogeneity (2)",
                           dep.var.labels.include = TRUE,
                           dep.var.labels = "Employment",
                           column.labels = c("Urban", "Rural"),
                           covariate.labels = "Treatment Effect",
                           type = "html",
                           out="GEGap_URA.html"
)

table_SN_GEGap <- stargazer(GEGap_result_DiD_S, GEGap_result_DiD_N,
                          dep.var.caption = "Heterogeneity (3)",
                          dep.var.labels.include = TRUE,
                          dep.var.labels = "Employment",
                          column.labels = c("South", "Nord"),
                          covariate.labels = "Treatment Effect",
                          type = "html",
                          out="GEGap_SN.html"
)


table_MNM_GEGap <- stargazer(GEGap_result_DiD_M, GEGap_result_DiD_NM,
                           dep.var.caption = "Heterogeneity (4)",
                           dep.var.labels.include = TRUE,
                           dep.var.labels = "Employment",
                           column.labels = c("Married", "Not M."),
                           covariate.labels = "Treatment Effect",
                           type = "html",
                           out="GEGap_MNM.html"
)

######
### Double Machine Learning with Random Forest (GEG) ===========================
######

  ### Prepare Data fpr DoubleML-Package ---

X1All_E
X1All_E_df <- X1All_E %>%  as.data.frame()
X1All_E_dt <- X1All_E_df %>% 
  mutate(log_hr_wage = GEGap$log_hr_wage,
         post_treat_fe = GEGap$post_treat_fe) %>%
  rename_with(~ tolower(gsub(":", "_", .x, fixed = TRUE))) %>% 
  as.data.table()
var_base_All_E <- head(names(X1All_E_dt), -2)

  ### Save Data for the Analyse
data_dml_GEGap_All <- DoubleMLData$new(X1All_E_dt,
                                 y_col = "log_hr_wage",
                                 d_cols = "post_treat_fe",
                                 x_cols = var_base_All_E)

  ### Random Forest Procedure according to DoubleML

randomForest_GEGap_All <- lrn("regr.ranger",
                          max.depth = 3,
                          mtry = 3,
                          min.node.size = 4)

randomForest_class_GEGap_All <- lrn("classif.ranger", 
                                    max.depth = 3,
                                    mtry = 3,
                                    min.node.size = 4)


dml_forest_GEGap_All = DoubleMLPLR$new(data_dml_GEGap_All,
                                     ml_l = randomForest_GEGap_All,
                                     ml_m = randomForest_class_GEGap_All,
                                     n_folds = 3)

dml_forest_GEGap_All$fit()
dml_forest_GEGap_All$summary()

  ### Save SE and Coefficient

RF_GEGap_Coef <- dml_forest_GEGap_All$coef
RF_GEGap_SE <- dml_forest_GEGap_All$se 

table_coef_GEGap <- rbind(RF_GEGap_Coef, RF_GEGap_SE)
rownames(table_coef_GEGap) <- c("Coefficient", "SE")
colnames(table_coef_GEGap) <- c("ATE on GEG")
table_coef_GEGap


table_coef_GEGap

################################################################################
############################## THE END #########################################
################################################################################