library(car)
library(emmeans)


# Cargar datos
df <- read.csv("C:\\Users\\Carlota\\Downloads\\data.csv", stringsAsFactors = TRUE)

df <- df[!(df$Structure == "2XJX" & df$Method == "CG2All" & df$Clash > 100), ]

df$Structure <- as.factor(df$Structure)
df$Method    <- as.factor(df$Method)

cat("Tamanios muestrales por celda:\n")
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
  cat("\n--- Tabla ANOVA (Tipo III) ---\n")
  print(anova_tabla)
  

  # Diagnostico de supuestos
  residuos <- residuals(modelo)
  
  cat("\n--- Test de normalidad de residuos (Shapiro-Wilk) ---\n")
  print(shapiro.test(residuos))
  
  cat("\n--- Test de homogeneidad de varianzas (Levene, por Method) ---\n")
  print(leveneTest(formula_str, data = data))
  
  cat("\n--- Test de homogeneidad de varianzas (Levene, por Structure) ---\n")
  print(leveneTest(as.formula(paste(response, "~ Structure")), data = data))
  
  # QQ-plot y residuos vs ajustados (se guardan como PNG)
  png(paste0("diagnostico_", response, ".png"), width = 900, height = 450)
  par(mfrow = c(1, 2))
  plot(modelo, which = 1)  # Residuos vs ajustados
  plot(modelo, which = 2)  # QQ-plot
  dev.off()
  
  # Post-hoc Tukey (si hay efectos significativos)
  cat("\n--- Post-hoc Tukey: Method ---\n")
  print(emmeans(modelo, pairwise ~ Method, adjust = "tukey")$contrasts)
  
  cat("\n--- Post-hoc Tukey: Structure ---\n")
  print(emmeans(modelo, pairwise ~ Structure, adjust = "tukey")$contrasts)
  
  cat("\n--- Post-hoc Tukey: interaccion Method:Structure ---\n")
  print(emmeans(modelo, pairwise ~ Method | Structure, adjust = "tukey")$contrasts)
  
  return(list(modelo = modelo, anova = anova_tabla))
}


# 3. Ejecutar para las 3 variables
resultado_clash    <- run_two_way_anova(df, "Clash",    "Clashscore")
resultado_ramaout  <- run_two_way_anova(df, "RamaOut",  "Ramachandran Outliers (%)")
resultado_rotaout  <- run_two_way_anova(df, "RotaOut",  "Sidechain Outliers (%)")


# 4. (Opcional) Analisis de sensibilidad: ANOVA sobre log(Clash+1)
#    util si el diagnostico de Clashscore muestra no-normalidad
#    fuerte por el outlier de 160.25 en 2XJX/CG2All
df$LogClash <- log(df$Clash + 1)
resultado_logclash <- run_two_way_anova(df, "LogClash", "log(Clashscore + 1)")

