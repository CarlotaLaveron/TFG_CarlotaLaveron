# ============================================================
# ANOVA de dos factores (Tipo III): Metodo x Estructura
# Variables: Clashscore (log-transformado), Ramachandran Outliers (%), Sidechain/Rota Outliers (%)
# ============================================================

# Instalar/cargar paquetes necesarios
if (!require(car)) install.packages("car")
library(car)

if (!require(emmeans)) install.packages("emmeans")
library(emmeans)

install.packages("nortest")


library(nortest)

# ------------------------------------------------------------
# 1. Cargar datos
# ------------------------------------------------------------
df <- read.csv("C:\\Users\\Carlota\\Desktop\\data_anova_full.csv", stringsAsFactors = TRUE)

df$Structure <- as.factor(df$Structure)
df$Method    <- as.factor(df$Method)

cat("Tamanios muestrales por celda:\n")
print(table(df$Structure, df$Method))

# ------------------------------------------------------------
# 2. Funcion para correr ANOVA tipo III + diagnosticos + Tukey
# ------------------------------------------------------------
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
  
  # --------------------------------------------------------
  # Diagnostico de supuestos
  # --------------------------------------------------------
  residuos <- residuals(modelo)
  
  cat("\n--- Test de normalidad de residuos (Shapiro-Wilk) ---\n")
  print(ad.test(residuos))
  
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
  
  # --------------------------------------------------------
  # Post-hoc Tukey (Method, Structure, e interaccion)
  # --------------------------------------------------------
  cat("\n--- Post-hoc Tukey: Method ---\n")
  print(emmeans(modelo, pairwise ~ Method, adjust = "tukey")$contrasts)
  
  cat("\n--- Post-hoc Tukey: Structure ---\n")
  print(emmeans(modelo, pairwise ~ Structure, adjust = "tukey")$contrasts)
  
  cat("\n--- Post-hoc Tukey: interaccion Method:Structure ---\n")
  print(emmeans(modelo, pairwise ~ Method | Structure, adjust = "tukey")$contrasts)
  
  return(list(modelo = modelo, anova = anova_tabla))
}

# ------------------------------------------------------------
# 3. Ejecutar para las 3 variables (Clashscore en log, las otras dos en bruto)
# ------------------------------------------------------------

df$LogClash <- log(df$Clash + 1)
df$RamaOut <- log(df$RamaOut + 1)
df$RotaOut <- log(df$RotaOut + 1)


resultado_logclash <- run_two_way_anova(df, "LogClash", "log(Clashscore + 1)")
resultado_ramaout   <- run_two_way_anova(df, "RamaOut",  "log(Ramachandran Outliers+1) (%)")
resultado_rotaout   <- run_two_way_anova(df, "RotaOut",  "log(Sidechain Outliers +1) (%)")

