library(car)
library(emmeans)


# Cargar datos
df <- read.csv("C:\\Users\\Carlota\\Desktop\\data_anova_full.csv", stringsAsFactors = TRUE)


df <- df[!(df$Structure == "2XJX" & df$Method == "CG2All" & df$Clash > 100), ]

df$Structure <- as.factor(df$Structure)
df$Method    <- as.factor(df$Method)

cat("Tamño muestrales por celda:\n")
print(table(df$Structure, df$Method))


# 2. Funcion para correr ANOVA tipo III + diagnosticos + Tukey
run_two_way_anova <- function(data, response, response_name) {
  
  cat("\n\n============================================================\n")
  cat("VARIABLE:", response_name, "\n")
  cat("============================================================\n")
  
  # Contrastes tipo "sum" necesarios para SS tipo III
  options(contrasts = c("contr.sum", "contr.poly"))
  
  formula_str <- as.formula(paste(response, "~ Method * Structure"))
  modelo <- lm(formula_str, data = data)
  
  # ANOVA tipo III
  anova_tabla <- Anova(modelo, type = "III")
  cat("Tabla ANOVA (Tipo III)\n")
  print(anova_tabla)
  

  # Diagnostico de supuestos
  residuos <- residuals(modelo)
  
  cat("\nTest de normalidad de residuos (Shapiro-Wilk)\n")
  print(shapiro.test(residuos))
  
  cat("\nTest de homogeneidad de varianzas (Levene, por Method)\n")
  print(leveneTest(formula_str, data = data))
  
  cat("\nTest de homogeneidad de varianzas (Levene, por Structure)n")
  print(leveneTest(as.formula(paste(response, "~ Structure")), data = data))
  
  # Post-hoc Tukey (si hay efectos significativos)
  cat("\n--- Post-hoc Tukey: Method ---\n")
  print(emmeans(modelo, pairwise ~ Method, adjust = "tukey")$contrasts)
  
  cat("\n--- Post-hoc Tukey: Structure ---\n")
  print(emmeans(modelo, pairwise ~ Structure, adjust = "tukey")$contrasts)
  
  cat("\n--- Post-hoc Tukey: interaccion Method:Structure ---\n")
  print(emmeans(modelo, pairwise ~ Method | Structure, adjust = "tukey")$contrasts)
  
  return(list(modelo = modelo, anova = anova_tabla))
}



resultado_clash    <- run_two_way_anova(df, "Clash",    "Clashscore")
resultado_ramaout  <- run_two_way_anova(df, "RamaOut",  "Ramachandran Outliers (%)")
resultado_rotaout  <- run_two_way_anova(df, "RotaOut",  "Sidechain Outliers (%)")


# 4. ANOVA sobre log(Clash+1)
df$LogClash <- log(df$Clash + 1)
resultado_logclash <- run_two_way_anova(df, "LogClash", "log(Clashscore + 1)")

