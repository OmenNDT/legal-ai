<?xml version="1.0" encoding="UTF-8"?>
<!-- MML2OMML.XSL - MathML to Office Math Markup Language (OMML) XSLT Transform
     Based on Microsoft's MML2OMML.XSL (simplified for common math elements)
     Converts MathML 2.0 → OMML for embedding in Word DOCX documents.
-->
<xsl:stylesheet version="1.0"
    xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
    xmlns:m="http://www.w3.org/1998/Math/MathML"
    xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    xmlns:mml="http://schemas.openxmlformats.org/officeDocument/2006/math"
    exclude-result-prefixes="m">

<xsl:output method="xml" indent="no" encoding="UTF-8"/>

<!-- Root math element -->
<xsl:template match="m:math">
  <mml:oMath>
    <xsl:apply-templates select="m:mrow/*|*[not(self::m:mrow)]"/>
  </mml:oMath>
</xsl:template>

<!-- mrow - just pass through children -->
<xsl:template match="m:mrow">
  <xsl:apply-templates/>
</xsl:template>

<!-- mi - identifier -->
<xsl:template match="m:mi">
  <mml:r>
    <mml:rPr>
      <mml:sty mml:val="i"/>
    </mml:rPr>
    <mml:t><xsl:value-of select="."/></mml:t>
  </mml:r>
</xsl:template>

<!-- mn - number -->
<xsl:template match="m:mn">
  <mml:r>
    <mml:t><xsl:value-of select="."/></mml:t>
  </mml:r>
</xsl:template>

<!-- mo - operator -->
<xsl:template match="m:mo">
  <mml:r>
    <xsl:if test="@stretchy='false'">
      <mml:rPr>
        <mml:sty mml:val="p"/>
      </mml:rPr>
    </xsl:if>
    <mml:t><xsl:value-of select="."/></mml:t>
  </mml:r>
</xsl:template>

<!-- mtext - text -->
<xsl:template match="m:mtext">
  <mml:r>
    <mml:rPr>
      <mml:sty mml:val="p"/>
    </mml:rPr>
    <mml:t><xsl:value-of select="."/></mml:t>
  </mml:r>
</xsl:template>

<!-- mfrac - fraction -->
<xsl:template match="m:mfrac">
  <mml:f>
    <mml:fPr>
      <mml:type mml:val="bar"/>
    </mml:fPr>
    <mml:num>
      <xsl:apply-templates select="child::*[1]"/>
    </mml:num>
    <mml:den>
      <xsl:apply-templates select="child::*[2]"/>
    </mml:den>
  </mml:f>
</xsl:template>

<!-- msub - subscript -->
<xsl:template match="m:msub">
  <mml:sSub>
    <mml:e>
      <xsl:apply-templates select="child::*[1]"/>
    </mml:e>
    <mml:sub>
      <xsl:apply-templates select="child::*[2]"/>
    </mml:sub>
  </mml:sSub>
</xsl:template>

<!-- msup - superscript -->
<xsl:template match="m:msup">
  <mml:sSup>
    <mml:e>
      <xsl:apply-templates select="child::*[1]"/>
    </mml:e>
    <mml:sup>
      <xsl:apply-templates select="child::*[2]"/>
    </mml:sup>
  </mml:sSup>
</xsl:template>

<!-- msubsup - subscript + superscript -->
<xsl:template match="m:msubsup">
  <mml:sSubSup>
    <mml:e>
      <xsl:apply-templates select="child::*[1]"/>
    </mml:e>
    <mml:sub>
      <xsl:apply-templates select="child::*[2]"/>
    </mml:sub>
    <mml:sup>
      <xsl:apply-templates select="child::*[3]"/>
    </mml:sup>
  </mml:sSubSup>
</xsl:template>

<!-- munder - underscript -->
<xsl:template match="m:munder">
  <mml:limLow>
    <mml:e>
      <xsl:apply-templates select="child::*[1]"/>
    </mml:e>
    <mml:lim>
      <xsl:apply-templates select="child::*[2]"/>
    </mml:lim>
  </mml:limLow>
</xsl:template>

<!-- mover - overscript -->
<xsl:template match="m:mover">
  <mml:limUpp>
    <mml:e>
      <xsl:apply-templates select="child::*[1]"/>
    </mml:e>
    <mml:lim>
      <xsl:apply-templates select="child::*[2]"/>
    </mml:lim>
  </mml:limUpp>
</xsl:template>

<!-- munderover -->
<xsl:template match="m:munderover">
  <mml:limLow>
    <mml:e>
      <mml:limUpp>
        <mml:e>
          <xsl:apply-templates select="child::*[1]"/>
        </mml:e>
        <mml:lim>
          <xsl:apply-templates select="child::*[3]"/>
        </mml:lim>
      </mml:limUpp>
    </mml:e>
    <mml:lim>
      <xsl:apply-templates select="child::*[2]"/>
    </mml:lim>
  </mml:limLow>
</xsl:template>

<!-- msqrt - square root -->
<xsl:template match="m:msqrt">
  <mml:rad>
    <mml:radPr>
      <mml:degHide mml:val="1"/>
    </mml:radPr>
    <mml:deg/>
    <mml:e>
      <xsl:apply-templates/>
    </mml:e>
  </mml:rad>
</xsl:template>

<!-- mroot - nth root -->
<xsl:template match="m:mroot">
  <mml:rad>
    <mml:deg>
      <xsl:apply-templates select="child::*[2]"/>
    </mml:deg>
    <mml:e>
      <xsl:apply-templates select="child::*[1]"/>
    </mml:e>
  </mml:rad>
</xsl:template>

<!-- mtable - matrix/table -->
<xsl:template match="m:mtable">
  <mml:m>
    <xsl:apply-templates/>
  </mml:m>
</xsl:template>

<!-- mtr - table row -->
<xsl:template match="m:mtr">
  <mml:mr>
    <xsl:apply-templates/>
  </mml:mr>
</xsl:template>

<!-- mtd - table cell -->
<xsl:template match="m:mtd">
  <mml:e>
    <xsl:apply-templates/>
  </mml:e>
</xsl:template>

<!-- mfenced - fenced (parentheses, brackets) -->
<xsl:template match="m:mfenced">
  <mml:d>
    <xsl:if test="@open">
      <mml:dPr>
        <mml:begChr mml:val="{@open}"/>
        <mml:endChr mml:val="{@close}"/>
      </mml:dPr>
    </xsl:if>
    <xsl:apply-templates/>
  </mml:d>
</xsl:template>

<!-- mspace - space -->
<xsl:template match="m:mspace">
  <mml:r>
    <mml:t>&#x0020;</mml:t>
  </mml:r>
</xsl:template>

<!-- mstyle - pass through -->
<xsl:template match="m:mstyle">
  <xsl:apply-templates/>
</xsl:template>

<!-- mpadded - pass through -->
<xsl:template match="m:mpadded">
  <xsl:apply-templates/>
</xsl:template>

<!-- mphantom -->
<xsl:template match="m:mphantom">
  <mml:r>
    <mml:rPr>
      <mml:sty mml:val="p"/>
    </mml:rPr>
    <mml:t><xsl:value-of select="."/></mml:t>
  </mml:r>
</xsl:template>

<!-- Catch-all: convert unknown elements to run -->
<xsl:template match="*">
  <mml:r>
    <mml:t><xsl:value-of select="."/></mml:t>
  </mml:r>
</xsl:template>

</xsl:stylesheet>