// Arduino "Hello World" fuer die Mechatronik Veranstaltung.

// Initialisierungsfunktion.
// Diese Funktion wird zu Beginn des Programms ausgefuehrt.
void setup() {
  // Pin 13 soll ein Ausgang sein.
  pinMode(13, OUTPUT);
}

// Diese Funktion wird periodisch ausgefuehrt.
void loop() {
  // Pin 13 auf einen hohen Logikpegel ziehen.
  digitalWrite(13, HIGH);
  // Warten (das Argument ist in ms gegeben).
  delay(500);
  // Pin 13 auf einen niedrigen Logikpegel ziehen.
  digitalWrite(13, LOW);
  // Warten (das Argument ist in ms gegeben).
  delay(500);
}
